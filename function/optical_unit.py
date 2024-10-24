import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.fft import fftshift, fft2, ifft2, ifftshift
from torchvision import transforms
from torch.autograd import Function
import torch.nn.functional as F


class Lens(nn.Module):
    def __init__(self, whole_dim, pixel_size, focal_length, wave_lambda):
        super(Lens, self).__init__()
        # basic parameters
        temp = np.arange(
            (-np.ceil((whole_dim - 1) / 2)), np.floor((whole_dim - 1) / 2) + 0.5
        )
        x = temp * pixel_size
        xx, yy = np.meshgrid(x, x)
        lens_function = np.exp(
            -1j * math.pi / wave_lambda / focal_length * (xx**2 + yy**2)
        )
        self.lens_function = torch.tensor(lens_function, dtype=torch.complex64).cuda()

    def forward(self, input_field):
        out = torch.mul(input_field, self.lens_function)
        return out


class AngSpecProp(nn.Module):
    def __init__(self, whole_dim, pixel_size, focal_length, wave_lambda):
        super(AngSpecProp, self).__init__()
        k = 2 * math.pi / wave_lambda  # optical wavevector
        df1 = 1 / (whole_dim * pixel_size)
        f = (
            np.arange(
                (-np.ceil((whole_dim - 1) / 2)), np.floor((whole_dim - 1) / 2) + 0.5
            )
            * df1
        )
        fxx, fyy = np.meshgrid(f, f)
        fsq = fxx**2 + fyy**2

        self.Q2 = torch.tensor(
            np.exp(-1j * (math.pi**2) * 2 * focal_length / k * fsq),
            dtype=torch.complex64,
        ).cuda()
        self.pixel_size = pixel_size
        self.df1 = df1

    def ft2(self, g, delta):
        return fftshift(fft2(ifftshift(g))) * (delta**2)

    def ift2(self, G, delta_f):
        N = G.shape[1]
        return ifftshift(ifft2(fftshift(G))) * ((N * delta_f) ** 2)

    def forward(self, input_field):
        # compute the propagated field
        Uout = self.ift2(self.Q2 * self.ft2(input_field, self.pixel_size), self.df1)
        return Uout


class ScaleSigner(Function):
    """take a real value x, output sign(x)*E(|x|)"""

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input) * torch.mean(torch.abs(input))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def scale_sign(input):
    return ScaleSigner.apply(input)


class Quantizer(Function):
    @staticmethod
    def forward(ctx, input, nbit):
        scale = 2**nbit - 1
        return torch.round(input * scale) / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def quantize(input, nbit):
    return Quantizer.apply(input, nbit)


def dorefa_w(w, nbit_w):
    if nbit_w == 1:
        w = scale_sign(w)
    else:
        w = torch.tanh(w)
        max_w = torch.max(torch.abs(w)).detach()
        w = w / 2 / max_w + 0.5
        w = 1.999 * quantize(w, nbit_w) - 1
    return w


def dorefa_a(input, nbit_a):
    # print(torch.clamp(0.1 * input, 0, 1))
    return quantize(torch.clamp(input, 0, 1), nbit_a)


class DMD(nn.Module):
    def __init__(self, whole_dim, phase_dim):
        super().__init__()
        self.whole_dim = whole_dim
        self.phase_dim = phase_dim
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(10.0), requires_grad=False)
        self.trans = Incoherent_Int2Complex()
        self.sensor = Sensor()
        self.mask = self.create_mask(whole_dim, phase_dim)

    def create_mask(self, whole_dim, phase_dim):
        pad_size = (whole_dim - phase_dim) // 2
        mask = torch.zeros((whole_dim, whole_dim))
        mask[pad_size : pad_size + phase_dim, pad_size : pad_size + phase_dim] = 1
        return mask

    def forward(self, x, insitu=False):
        # print(x.shape)
        if not insitu:
            modulus_squared = self.sensor(x)
            # print(modulus_squared.max(), modulus_squared.min())
            # modulus_squared = self.conv(modulus_squared.unsqueeze(1)).squeeze(1)
            # modulus_squared = dorefa_a(modulus_squared, 8)
        else:
            # x = x **2
            # x = torch.tanh(x)
            modulus_squared = x
        # print(modulus_squared.device)
        # modulus_squared = self.ln(modulus_squared)
        mask = self.mask.to(x.device)
        # modulus_squared = modulus_squared * mask
        # print(modulus_squared.max(), modulus_squared.min())

        I_th = torch.mean(modulus_squared, dim=(-2, -1), keepdim=True)
        x = torch.sigmoid(self.beta * (modulus_squared - self.alpha * I_th))
        y = dorefa_a(x, 1)

        x = self.trans(y)

        x_real = x.real * mask
        x_imag = x.imag * mask
        x = torch.complex(x_real, x_imag)

        return x


class PhaseMask(nn.Module):
    def __init__(self, whole_dim, phase_dim, phase=None):
        super(PhaseMask, self).__init__()
        self.whole_dim = whole_dim
        phase = (
            torch.randn(1, phase_dim, phase_dim, dtype=torch.float32)
            if phase is None
            else torch.tensor(phase, dtype=torch.float32)
        )
        self.w_p = nn.Parameter(phase)
        pad_size = (whole_dim - phase_dim) // 2
        self.paddings = (pad_size, pad_size, pad_size, pad_size)
        self.init_weights()

    # kaiming init
    def init_weights(self):
        nn.init.kaiming_uniform_(self.w_p, a=math.sqrt(5))

    def forward(self, input_field):
        mask_phase = (dorefa_w(self.w_p, 8)) * math.pi
        mask_whole = F.pad(
            torch.complex(torch.cos(mask_phase), torch.sin(mask_phase)), self.paddings
        ).cuda()
        output_field = torch.mul(input_field, mask_whole)
        return output_field


class NonLinear_Int2Phase_for_DMD(nn.Module):
    def __init__(self):
        super(NonLinear_Int2Phase_for_DMD, self).__init__()

    def forward(self, input_field):
        phase = input_field * 1.999 * math.pi
        print(phase)
        phase = torch.complex(torch.cos(phase), torch.sin(phase)).cuda()
        return phase


class NonLinear_Int2Phase(nn.Module):
    def __init__(self):
        super(NonLinear_Int2Phase, self).__init__()

    def forward(self, input_field):
        phase = input_field * 1.999 * math.pi
        phase = torch.complex(torch.cos(phase), torch.sin(phase)).cuda()
        return phase


class Incoherent_Int2Complex(nn.Module):
    def __init__(self):
        super(Incoherent_Int2Complex, self).__init__()

    def forward(self, input_field):
        # print(input_field.dtype)
        x = torch.complex(
            input_field,
            torch.zeros(
                input_field.shape, device=input_field.device, dtype=input_field.dtype
            ),
        ).cuda()
        return x


class Sensor(nn.Module):
    def __init__(self):
        super(Sensor, self).__init__()

    def forward(self, input_field):
        x = torch.square(torch.real(input_field)) + torch.square(
            torch.imag(input_field)
        )
        return torch.tanh(x)


def test_sensor():
    sensor = Sensor()
    x = torch.randn(1, 100, 100)
    x = dorefa_w(x, 8)
    print(torch.unique(x))


if __name__ == "__main__":
    test_sensor()
