def _richardson_lucy_deblur(self, img, iterations=10, psf_size=5):
    """
    Richardson-Lucy 디콘볼루션 (초점 불량 복원)
    
    가우시안 PSF(Point Spread Function)를 가정하여 흐릿함을 제거합니다.
    """
    if not SCIPY_AVAILABLE:
        return img
    
    try:
        # PSF 생성 (가우시안 블러 커널)
        psf = np.zeros((psf_size, psf_size))
        psf[psf_size//2, psf_size//2] = 1
        psf = gaussian_filter(psf, sigma=1.5)
        psf /= psf.sum()
        
        # Richardson-Lucy 반복
        img_float = img.astype(np.float64) / 255.0
        img_float = np.maximum(img_float, 1e-10)  # 0 방지
        
        estimated = img_float.copy()
        for _ in range(iterations):
            # 순방향 컨볼루션
            reblurred = convolve2d(estimated, psf, mode='same', boundary='symm')
            reblurred = np.maximum(reblurred, 1e-10)
            
            # 역전파
            ratio = img_float / reblurred
            correction = convolve2d(ratio, psf[::-1, ::-1], mode='same', boundary='symm')
            estimated *= correction
            estimated = np.maximum(estimated, 1e-10)
        
        # 0-255 범위로 변환
        result = np.clip(estimated * 255, 0, 255).astype(np.uint8)
        return result
    except Exception as e:
        logger.warning(f"[QR] Richardson-Lucy deblur failed: {e}")
        return img
