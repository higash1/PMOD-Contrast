import torch
from torch import nn
from torch import Tensor
from typing import Tuple

class DDPM(nn.Module):
    def __init__(self, T, device):
        super().__init__()
        self.device = device
        self.T = T
        # β1 and βt はオリジナルの ddpm reportに記載されている値を採用します
        self.beta_1:float = 1e-4
        self.beta_T:float = 0.02

        self.betas: Tensor = torch.linspace(self.beta_1, self.beta_T, T, device=device)
        
        self.alphas:float = 1.0 - self.betas
        
        self.alpha_bars:Tensor = torch.cumprod(self.alphas, dim=0)

    def diffusion_process(self, x0, t=None):
        if t is None:
            t: Tensor = torch.randint(low=1, high=self.T, size=(x0.shape[0],), device=self.device)
        noise: Tensor = torch.randn_like(x0, device=self.device)
        alpha_bar:Tensor = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        # 1000ステップの順方向の拡散過程を計算します
        xt:Tensor = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
        return xt
    
    def extract_nonfixed_img(self, original_img, mask):
        original_img = original_img * mask
        return original_img
    
    def concat_img(self, noise_img, original_img, mask):
        noise_img = noise_img * mask
        original_img = original_img * (1 - mask)
        return noise_img + original_img
    
    def generate_mask_expanddim(self, label_img):
        # b, h, w
        mask_img = torch.zeros_like(label_img)
        for id in self.dynamic_list:
            mask_img[torch.where(label_img == id)] = 1
        return mask_img.unsqueeze(1).repeat(1, 3, 1, 1)
    
    def set_dynamic_list(self, dynamic_list):
        self.dynamic_list = dynamic_list
    
# if __name__ == "__main__":
#     x0 = dataset[3]
#     # バッチ次元がないので、追加します
#     x0 = x0.unsqueeze(0)

#     ddpm = DDPM(1000, "cpu")
#     xt, t, noise = ddpm.diffusion_process(x0)
#     print(f"{xt.shape=}", f"\n{t.shape=}", f"\n{noise.shape=}")

#     # 以下は描画のための関数です
#     fig, ax = plt.subplots(1,3)
#     ax[0].imshow(1. - x0[0, 0, :, :], cmap="gray")
#     ax[1].imshow(1. - xt[0, 0, :, :], cmap="gray")
#     ax[2].imshow(1. - noise[0, 0, :, :], cmap="gray")
#     plt.show()
#     plt.clf()
#     plt.close()