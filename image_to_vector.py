"""
이미지를 딥러닝에 사용할 수 있는 벡터 형태로 변환하는 모듈
"""
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional
import os


class ImageToVector:
    """이미지를 딥러닝 벡터로 변환하는 클래스"""
    
    def __init__(self, target_size: Optional[Tuple[int, int]] = None, 
                 normalize: bool = True, 
                 grayscale: bool = False):
        """
        Parameters:
        -----------
        target_size : Tuple[int, int], optional
            변환할 이미지 크기 (width, height). None이면 원본 크기 유지
        normalize : bool
            True면 픽셀 값을 0-1 범위로 정규화, False면 0-255 범위 유지
        grayscale : bool
            True면 그레이스케일로 변환, False면 컬러 유지
        """
        self.target_size = target_size
        self.normalize = normalize
        self.grayscale = grayscale
    
    def load_image(self, image_path: str) -> Image.Image:
        """
        이미지 파일을 로드합니다.
        
        Parameters:
        -----------
        image_path : str
            이미지 파일 경로
        
        Returns:
        --------
        PIL.Image.Image
            로드된 이미지 객체
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        
        image = Image.open(image_path)
        return image
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        이미지를 전처리하여 NumPy 배열로 변환합니다.
        
        Parameters:
        -----------
        image : PIL.Image.Image
            전처리할 이미지
        
        Returns:
        --------
        np.ndarray
            전처리된 이미지 배열 (H, W, C) 또는 (H, W) 형태
        """
        # 그레이스케일 변환
        if self.grayscale:
            image = image.convert('L')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 리사이즈
        if self.target_size:
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # NumPy 배열로 변환
        img_array = np.array(image)
        
        # 정규화
        if self.normalize:
            img_array = img_array.astype(np.float32) / 255.0
        
        return img_array
    
    def to_vector(self, image_path: str) -> np.ndarray:
        """
        이미지 파일을 벡터 형태로 변환합니다.
        
        Parameters:
        -----------
        image_path : str
            이미지 파일 경로
        
        Returns:
        --------
        np.ndarray
            벡터 형태의 이미지 데이터
        """
        image = self.load_image(image_path)
        vector = self.preprocess_image(image)
        return vector
    
    def to_flatten_vector(self, image_path: str) -> np.ndarray:
        """
        이미지를 1차원 벡터로 변환합니다.
        
        Parameters:
        -----------
        image_path : str
            이미지 파일 경로
        
        Returns:
        --------
        np.ndarray
            1차원 벡터 형태의 이미지 데이터
        """
        vector = self.to_vector(image_path)
        return vector.flatten()
    
    def batch_process(self, image_paths: list) -> np.ndarray:
        """
        여러 이미지를 배치로 처리합니다.
        
        Parameters:
        -----------
        image_paths : list
            이미지 파일 경로 리스트
        
        Returns:
        --------
        np.ndarray
            배치 형태의 벡터 데이터 (N, H, W, C) 또는 (N, H, W)
        """
        vectors = []
        for path in image_paths:
            vector = self.to_vector(path)
            vectors.append(vector)
        
        return np.array(vectors)


def convert_image_to_vector(image_path: str, 
                           target_size: Optional[Tuple[int, int]] = None,
                           normalize: bool = True,
                           grayscale: bool = False,
                           flatten: bool = False) -> np.ndarray:
    """
    이미지를 딥러닝 벡터로 변환하는 편의 함수
    
    Parameters:
    -----------
    image_path : str
        이미지 파일 경로
    target_size : Tuple[int, int], optional
        변환할 이미지 크기 (width, height)
    normalize : bool
        픽셀 값을 0-1 범위로 정규화할지 여부
    grayscale : bool
        그레이스케일로 변환할지 여부
    flatten : bool
        True면 1차원 벡터로 변환
    
    Returns:
    --------
    np.ndarray
        벡터 형태의 이미지 데이터
    """
    converter = ImageToVector(
        target_size=target_size,
        normalize=normalize,
        grayscale=grayscale
    )
    
    if flatten:
        return converter.to_flatten_vector(image_path)
    else:
        return converter.to_vector(image_path)


def main():
    """메인 함수 - 사용 예제"""
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python image_to_vector.py <이미지_경로> [옵션]")
        print("\n옵션:")
        print("  --size WIDTH HEIGHT  : 이미지 크기 지정 (예: --size 224 224)")
        print("  --no-normalize       : 정규화 비활성화")
        print("  --grayscale          : 그레이스케일 변환")
        print("  --flatten            : 1차원 벡터로 변환")
        print("\n예제:")
        print("  python image_to_vector.py image.jpg")
        print("  python image_to_vector.py image.jpg --size 224 224 --grayscale")
        return
    
    image_path = sys.argv[1]
    
    # 옵션 파싱
    target_size = None
    normalize = True
    grayscale = False
    flatten = False
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--size' and i + 2 < len(sys.argv):
            target_size = (int(sys.argv[i+1]), int(sys.argv[i+2]))
            i += 3
        elif sys.argv[i] == '--no-normalize':
            normalize = False
            i += 1
        elif sys.argv[i] == '--grayscale':
            grayscale = True
            i += 1
        elif sys.argv[i] == '--flatten':
            flatten = True
            i += 1
        else:
            i += 1
    
    try:
        print(f"이미지 로드 중: {image_path}")
        vector = convert_image_to_vector(
            image_path,
            target_size=target_size,
            normalize=normalize,
            grayscale=grayscale,
            flatten=flatten
        )
        
        print(f"\n변환 완료!")
        print(f"벡터 형태: {vector.shape}")
        print(f"데이터 타입: {vector.dtype}")
        print(f"값 범위: {vector.min():.3f} ~ {vector.max():.3f}")
        print(f"\n벡터 데이터 샘플 (처음 10개):")
        print(vector.flatten()[:10])
        
    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    main()

