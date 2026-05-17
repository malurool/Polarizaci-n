import os
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Function
from spectral.io import envi

V_mean = .0
T_mean = 300
emissivity_mean = 0.9
d_mean = 120

V_std = 1
T_std = 10
emissivity_std = 10
d_std = 1000

class BlackbodyFunction(Function):
    @staticmethod
    def forward(ctx, lambda_um, T):
        """
        Forward pass for the blackbody function.
        """
        ctx.save_for_backward(lambda_um, T)

        # Convert wavelength from micrometers to meters
        lambda_m = lambda_um * 1e-6

        h = 6.63e-34  # Planck's constant (J*s)
        c = 3e8  # Speed of light (m/s)
        k_B = 1.38e-23  # Boltzmann constant (J/K)

        # Calculate exponent term and clamp to avoid overflow
        exponent = h * c / (lambda_m * k_B * T)
        exponent = torch.clamp(exponent, max=700)  # Clamp to prevent overflow
        exp_term = torch.exp(exponent)

        # Calculate microflicks (spectral radiance)
        microflicks = 1e-4 * (2 * h * c ** 2 / (lambda_m ** 5 * (exp_term - 1)))

        return microflicks

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the blackbody function.
        """
        lambda_um, T = ctx.saved_tensors
        lambda_m = lambda_um * 1e-6

        h = 6.63e-34  # Planck's constant (J*s)
        c = 3e8  # Speed of light (m/s)
        k_B = 1.38e-23  # Boltzmann constant (J/K)

        # Calculate exponent term and clamp to avoid overflow
        exponent = h * c / (lambda_m * k_B * T)
        exponent = torch.clamp(exponent, max=700)  # Clamp to prevent overflow
        exp_term = torch.exp(exponent)

        # Compute the gradient with respect to temperature T
        # dT is the partial derivative of the blackbody function w.r.t T
        dI_dT = (2 * h * c ** 2 / lambda_m ** 5) * (exp_term * exponent / (T ** 2 * (exp_term - 1) ** 2))

        # Multiply by the incoming gradient from the next layer
        dT = dI_dT * grad_output

        return None, dT

def read_data_file(directory, filename):
    """
    Reads a .mat or .npz file from the given directory.

    Parameters:
    directory (str): The directory where the .mat file is located.
    filename (str): The name of the .mat file.

    Returns:
    dict: A dictionary containing the variables in the .mat file.
    """
    filepath = os.path.join(directory, filename)

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No such file: '{filepath}'")

    if filepath.lower().endswith(".npz"):
        with np.load(filepath, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    mat_contents = scipy.io.loadmat(filepath)
    return mat_contents


def read_hdr_cube(hdr_path):
    img = envi.open(hdr_path)
    cube = np.asarray(img.load())
    wavelength = img.metadata.get("wavelength")
    if wavelength is not None:
        wavelength = np.array([float(value) for value in wavelength])
        if np.nanmax(wavelength) > 100:
            wavelength = wavelength / 1000.0
    return cube, wavelength


def load_data(HSI_directory, filename, downwelling_flag = True):
    data_dir = os.getenv("DATA_DIR", HSI_directory)

    # Load HSI
    if filename.lower().endswith(".hdr"):
        hdr_path = filename if os.path.isabs(filename) else os.path.join(HSI_directory, filename)
        HSI, wavelength = read_hdr_cube(hdr_path)
        T_air = float(os.getenv("T_AIR", "289.7"))
        if wavelength is None:
            wavelength = read_data_file(data_dir, "lambda.npz")["lambda"]
    else:
        HSI = read_data_file(HSI_directory, filename)
        wavelength = HSI['lambda']
        T_air = HSI['T_air']
        HSI = HSI['meas']

    # Load Downwelling Radiances
    dw_r = read_data_file(data_dir, 'I_downwelling_res.npz')
    dw_r = dw_r['I_downwelling_res']

    # Load Attenuation Profile
    attenuation = read_data_file(data_dir, 'attenuation.npz')
    attenuation = attenuation['attenuation']

    HSI = HSI[:, :, 0:247]
    wavelength = np.atleast_2d(wavelength.ravel()[0:247])
    attenuation = np.atleast_2d(attenuation.ravel()[0:247])
    dw_r = dw_r[0:247, :]

    [N, M, K] = HSI.shape
    L = dw_r.shape[1]

    HSI = np.reshape(HSI, (N, M, K, 1))
    wavelength = np.reshape(wavelength, (1, 1, K, 1))
    dw_r = np.reshape(dw_r, (1, 1, K, L))
    attenuation = np.reshape(attenuation, (1, 1, K, 1))

    if not downwelling_flag:
        dw_r = np.zeros_like(dw_r)
    
    return HSI, wavelength, dw_r, attenuation, T_air

def standardize_data(data, mean, std):
    return (data - mean) / std

def destandardize_data(data, mean, std):
    return (data * std) + mean


def compute_incident_light(V, wavelength, dw_r, T_env):
    """
    Calculate the incident light based on the blackbody radiation and provided parameters.

    Parameters:
    V (torch.Tensor): Light intensity with shape (batch_size, height, width, num_wavelengths).
    wavelength (torch.Tensor): Wavelengths in micrometers with shape (num_wavelengths,).
    dw_r (torch.Tensor): Angular direction with shape (batch_size, height, width, num_wavelengths).
    T_env (float): Environmental temperature in Kelvin.

    Returns:
    torch.Tensor: Incident light with shape (batch_size, height, width, 1).
    """
    # Calculate the blackbody spectral radiance
    bb_env = BlackbodyFunction.apply(wavelength, T_env)

    # Concatenate along the last dimension to match dw_r
    light_dir = torch.cat((dw_r, bb_env), dim=3)  # Shape: (batch_size, height, width, num_wavelengths + 1)

    # Calculate the incident light by summing over the last dimension
    incident_light = torch.sum(V * light_dir, dim=3, keepdim=True)

    return incident_light


def forward_model(wavelength, T, emissivity, attenuation, d, T_air, incident_light):

    # Compute the object emission
    obj_emission = BlackbodyFunction.apply(wavelength, T) * emissivity

    # Compute the object reflection
    obj_reflection = incident_light * (1 - emissivity)

    # Compute the attenuation factor
    tau = 10 ** (-attenuation * d / 10)

    # Compute the sensor radiance
    sensor_radiance = tau * (obj_emission + obj_reflection) + (1 - tau) * BlackbodyFunction.apply(wavelength, T_air)
    return sensor_radiance


def l2_loss(measured, model_output):
    return F.mse_loss(model_output, measured)


def tikhonov_regularization(emissivity, alpha=1.0):
    """
    Compute the Tikhonov regularization loss by taking the circular difference along axis 2,
    squaring it, and summing the results.

    Parameters:
    - emissivity (Tensor): The emissivity values with shape [M, N, K, 1].
    - alpha (float): Regularization parameter (lambda).

    Returns:
    - Tensor: Regularization loss.
    """
    # Compute the forward difference along axis 2
    diff = emissivity[:, :, 1:, :] - emissivity[:, :, :-1, :]

    # Compute the circular difference between the last and first element along axis 2
    circular_diff = emissivity[:, :, 0, :] - emissivity[:, :, -1, :]

    # Concatenate forward differences with circular difference
    diff = torch.cat([diff, circular_diff.unsqueeze(2)], dim=2)

    # Square the differences
    squared_diff = diff ** 2

    # Sum the squared differences
    reg_term = torch.mean(squared_diff)

    return alpha * reg_term

def total_variation_regularization(d, alpha=1.0):
    """
    Compute the Total Variation (TV) regularization loss for the parameter d using absolute differences.

    Parameters:
    - d (Tensor): The parameter d with shape [M, N, 1, 1].
    - alpha (float): Regularization parameter (lambda).

    Returns:
    - Tensor: Regularization loss.
    """
    # Compute the forward difference along dim=0
    diff_d0 = d[:, 1:, :, :] - d[:, :-1, :, :]

    # Compute the forward difference along dim=1
    diff_d1 = d[1:, :, :, :] - d[:-1, :, :, :]

    # Compute the absolute differences
    abs_diff_d0 = torch.abs(diff_d0)
    abs_diff_d1 = torch.abs(diff_d1)

    # Sum the absolute differences
    reg_term_d0 = torch.mean(abs_diff_d0)
    reg_term_d1 = torch.mean(abs_diff_d1)

    # Total variation regularization loss
    reg_term = alpha * (reg_term_d0 + reg_term_d1)

    return reg_term


def total_loss(measured, model_output, emissivity, d, alpha=1.0, alpha_2=1.0):

    l2 = l2_loss(measured, model_output)
    reg = tikhonov_regularization(emissivity, alpha)
    reg2 = total_variation_regularization(d, alpha_2)
    return l2 + reg + reg2


def solve(wavelength, dw_r, T_env, measured, attenuation, T_air, num_iterations=100, lr=0.1, alpha=1.0, alpha_2 = 0, start_point = None, optimizer_type='SGD'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move tensors to GPU
    wavelength = wavelength.to(device)
    dw_r = dw_r.to(device)
    T_env = torch.tensor(T_env, dtype=torch.float32, device=device)
    T_air = torch.tensor(T_air, dtype=torch.float32, device=device)
    measured = measured.to(device)
    attenuation = attenuation.to(device, dtype=torch.float32)

    if start_point:
        V = torch.tensor(start_point['V'], dtype=torch.float32, device=device, requires_grad=True)
        T = torch.tensor(start_point['T'], dtype=torch.float32, device=device, requires_grad=True)
        emissivity = torch.tensor(start_point['emissivity'], dtype=torch.float32, device=device, requires_grad=True)
        d = torch.tensor(start_point['d'], dtype=torch.float32, device=device, requires_grad=True)
        print(V.shape)
        print(T.shape)
        print(emissivity.shape)
        print(d.shape)

        print("Initialized from starting point")
    else:

        # # Initialize parameters
        # V = torch.full((measured.shape[0], measured.shape[1], 1, 7), 1e-3, dtype=torch.float32, device=device,
        #            requires_grad=True)  # Estimated V
        # T = torch.full((measured.shape[0], measured.shape[1], 1, 1), 295, dtype=torch.float32, device=device,
        #            requires_grad=True)  # Temperature
        # emissivity = torch.full((measured.shape[0], measured.shape[1], measured.shape[2], 1), 1, dtype=torch.float32, device=device,
        #            requires_grad=True)  # Temperature
        # d = torch.full((measured.shape[0], measured.shape[1], 1, 1), 200, dtype=torch.float32, device=device, requires_grad=True)

        # Initialize parameters
        V = torch.full((measured.shape[0], measured.shape[1], 1, 11), 0, dtype=torch.float32, device=device,
                       requires_grad=True)  # Estimated V
        T = torch.full((measured.shape[0], measured.shape[1], 1, 1), 0, dtype=torch.float32, device=device,
                       requires_grad=True)  # Temperature
        emissivity = torch.full((measured.shape[0], measured.shape[1], measured.shape[2], 1), 0, dtype=torch.float32,
                                device=device,
                                requires_grad=True)  # Temperature
        d = torch.full((measured.shape[0], measured.shape[1], 1, 1), 0, dtype=torch.float32, device=device,
                       requires_grad=True)

        print("No starting point")


    # Optimizer
    if optimizer_type == 'SGD':
        optimizer = optim.SGD([V, T, emissivity, d], lr=lr)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam([V, T, emissivity, d], lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # Lists to store losses
    total_losses = []
    l2_losses = []
    reg_losses = []

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        V_r = destandardize_data(V, V_mean, V_std)
        T_r = destandardize_data(T, T_mean, T_std)
        d_r = destandardize_data(d, d_mean, d_std)
        emissivity_r = destandardize_data(emissivity, emissivity_mean, emissivity_std)

        # Compute incident light
        incident_light = compute_incident_light(V_r, wavelength, dw_r, T_env)

        # Compute model output
        model_output = forward_model(wavelength, T_r, emissivity_r, attenuation, d_r, T_air, incident_light)

        # Compute losses
        #loss = total_loss(measured, model_output, emissivity_r, d_r, alpha, alpha_2=5e-8)
        loss = total_loss(measured, model_output, emissivity_r, d_r, alpha, alpha_2)
        l2_loss_value = l2_loss(measured, model_output).item()
        reg_loss_value = tikhonov_regularization(emissivity, alpha).item()

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_([emissivity], max_norm=1.0)

        # Update parameters
        optimizer.step()

        # # Apply constraints
        with torch.no_grad():

            # Ensure V is positive
            V.data = torch.relu(destandardize_data(V.data, V_mean, V_std))

            # Normalize V to ensure sum along the third axis is less than 1
            V_sum = V.data.sum(dim=3, keepdim=True)
            V.data = torch.where(V_sum > 1, V.data / V_sum * 0.99, V.data)  # Scale to ensure sum < 1
            V.data = standardize_data(V.data, V_mean, V_std)

            # Clamp T to avoid NaNs
            T.data = destandardize_data(T.data, T_mean, T_std)
            T.data = torch.clamp(T.data, min=0.0, max=400)  # Adjust the range as needed
            T.data = standardize_data(T.data, T_mean, T_std)

            # Clamp d to avoid NaNs
            d.data = destandardize_data(d.data, d_mean, d_std)
            d.data = torch.clamp(d.data, min=0.0, max= 200)  # Adjust the range as needed
            d.data = standardize_data(d.data, d_mean, d_std)

            # Emiss d to avoid NaNs
            emissivity.data =  destandardize_data(emissivity.data, emissivity_mean, emissivity_std)
            emissivity.data = torch.clamp(emissivity.data, min=0.0, max=1)  # Adjust the range as needed
            emissivity.data = standardize_data(emissivity.data, emissivity_mean, emissivity_std)

        #Store losses
        total_losses.append(loss.item())
        l2_losses.append(l2_loss_value)
        reg_losses.append(reg_loss_value)

        if iteration % 1000 == 0:
            print(f"Iteration {iteration}/{num_iterations}, Loss: {loss.item()}")

    # Save loss curve
    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Figures', 'Training')
    os.makedirs(figures_dir, exist_ok=True)
    fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
    ax_loss.semilogy(total_losses, label='Total Loss')
    ax_loss.semilogy(l2_losses, label='L2 Loss')
    ax_loss.semilogy(reg_losses, label='Regularization')
    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training Loss')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    fig_loss.tight_layout()
    fig_loss.savefig(os.path.join(figures_dir, 'loss.pdf'), bbox_inches='tight')
    fig_loss.savefig(os.path.join(figures_dir, 'loss.jpg'), bbox_inches='tight', dpi=150)
    plt.close(fig_loss)

    return V_r.detach().cpu().numpy(), T_r.detach().cpu().numpy(), emissivity_r.detach().cpu().numpy(), d_r.detach().cpu().numpy()

def solve_full_scene(HSI_directory, filename, downwelling_flag=True, chunk_size=64, lr=1e-2, num_iterations=100000, emiss_reg=1e7, TV_reg=0):
    HSI, wavelength, dw_r, attenuation, T_air = load_data(HSI_directory, filename, downwelling_flag=downwelling_flag)
    HSI = HSI[:, -256:, :, :]

    # Crop the first and second dimensions to be multiples of the chunk size
    crop_size_0 = (HSI.shape[0] // chunk_size) * chunk_size
    crop_size_1 = (HSI.shape[1] // chunk_size) * chunk_size
    HSI = HSI[:crop_size_0, :crop_size_1, :, :]

    # Converting the data to PyTorch tensors
    HSI = torch.from_numpy(np.ascontiguousarray(HSI)).float()
    wavelength = torch.from_numpy(np.ascontiguousarray(wavelength)).float()
    attenuation = torch.from_numpy(np.ascontiguousarray(attenuation)).float()
    dw_r = torch.from_numpy(np.ascontiguousarray(dw_r)).float()
    

    T_env = 0  # Environmental temperature (Basically ignore this)

    # Initialize tensors to store the results
    V_full = torch.zeros(HSI.shape[0], HSI.shape[1], 1, 11)
    T_full = torch.zeros(HSI.shape[0], HSI.shape[1], 1, 1)
    emissivity_full = torch.zeros(HSI.shape[0], HSI.shape[1], HSI.shape[2], 1)
    d_full = torch.zeros(HSI.shape[0], HSI.shape[1], 1, 1)

    # Process data in chunks along the first and second dimensions
    for i in range(0, HSI.shape[0], chunk_size):
        for j in range(0, HSI.shape[1], chunk_size):
            # Define the chunk
            HSI_chunk = HSI[i:i + chunk_size, j:j + chunk_size, :, :]

            # Solve the optimization problem for the chunk
            if downwelling_flag:
                V, T, emissivity, d = solve(wavelength, dw_r, T_env, HSI_chunk, attenuation, num_iterations=num_iterations,
                                         T_air=T_air, lr=lr, alpha=emiss_reg, alpha_2=TV_reg, start_point=None, optimizer_type='SGD')
            else:
                V, T, emissivity, d = solve(wavelength, dw_r, T_env, HSI_chunk, attenuation, num_iterations=num_iterations,
                                         T_air=T_air, lr=lr, alpha=emiss_reg, alpha_2=TV_reg, start_point=None, optimizer_type='Adam')
            # Convert the results to PyTorch tensors
            V = torch.from_numpy(V).float()
            T = torch.from_numpy(T).float()
            emissivity = torch.from_numpy(emissivity).float()
            d = torch.from_numpy(d).float()

            # Store the results in the full tensors
            V_full[i:i + chunk_size, j:j + chunk_size, :, :] = V
            T_full[i:i + chunk_size, j:j + chunk_size, :, :] = T
            emissivity_full[i:i + chunk_size, j:j + chunk_size, :, :] = emissivity
            d_full[i:i + chunk_size, j:j + chunk_size, :, :] = d

    V_out = V_full.cpu().numpy()
    T_out = T_full.cpu().numpy()
    emissivity_out = emissivity_full.cpu().numpy()
    d_out = d_full.cpu().numpy()

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(repo_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    scene_name = os.path.splitext(os.path.basename(filename))[0]
    suffix = 'downwelling' if downwelling_flag else 'no_downwelling'
    savename = os.path.join(results_dir, f"{scene_name}_{suffix}.npz")
    np.savez(savename, V=V_out, T=T_out, emissivity=emissivity_out, d=d_out)
    print(f"Saved: {savename}")

    return V_out, T_out, emissivity_out, d_out

def main():
    hdr_path = os.getenv("HDR_PATH", "")
    HSI_directory = os.getenv("HSI_DIR", "")
    filename = os.getenv("HSI_FILENAME", "")
    if hdr_path and not HSI_directory:
        HSI_directory = os.path.dirname(hdr_path)
    if hdr_path:
        filename = hdr_path
    lr = .01 # Optimizer learning rate
    emiss_reg = 1e7 # Emissivity smoothness regularization parameter
    TV_reg = 1e-4 # TV regularization parameter for distance d
    downwelling_flag = os.getenv("DOWNWELLING_FLAG", "True").strip().lower() != "false"
    chunk_size = 128 # Pixel chunk size for processing in GPU
    if downwelling_flag:
        num_iterations = 100000 # With downwelling requires more iterations
    else:
        num_iterations = 20000 # Without downwelling converges faster
        lr = .0005 # Optimizer learning rate

    V, T, emissivity, d = solve_full_scene(HSI_directory, filename, downwelling_flag=downwelling_flag, chunk_size=chunk_size, lr=lr, num_iterations=num_iterations, emiss_reg=emiss_reg, TV_reg=TV_reg)
    return V, T, emissivity, d

if __name__ == "__main__":
    main()
