"""
Jovimetrix - http://www.github.com/amorano/jovimetrix
Work in Progress
"""

import torch

from Jovimetrix import deep_merge_dict, tensor2cv, cv2tensor, cv2mask, \
    JOVBaseNode, Logger, IT_PIXELS

# =============================================================================
# === 😱 JUNK AREA 😱 ===
# =============================================================================

class AkashicNode(JOVBaseNode):
    NAME = "AKASHIC (JOV) 📓"
    CATEGORY = "JOVIMETRIX 🔺🟩🔵/UTILITY"
    DESCRIPTION = ""
    RETURN_TYPES = ('MASK',)
    RETURN_NAMES = ("🦄",)
    OUTPUT_IS_LIST = (False,)
    SORT = 0
    POST = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        d = {"required": {
        }}
        return deep_merge_dict(IT_PIXELS, d)

    def run(self, image ) -> tuple[torch.Tensor, torch.Tensor]:
        image = tensor2cv(image)

        return (cv2tensor(image),
                cv2mask(image))

# =============================================================================
# === TESTING ===
# =============================================================================

if __name__ == "__main__":
    pass
