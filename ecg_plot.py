import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # Windows에서 한글 폰트 설정
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

def generate_ecg_signal(duration=10, sampling_rate=250):
    """
    심전도 신호를 생성합니다.
    
    Parameters:
    -----------
    duration : float
        신호의 지속 시간 (초)
    sampling_rate : int
        샘플링 주파수 (Hz)
    
    Returns:
    --------
    t : numpy array
        시간 배열
    ecg : numpy array
        심전도 신호
    """
    # 시간 배열 생성
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # 기본 심박수 (BPM)
    heart_rate = 72  # 분당 72회
    heart_period = 60.0 / heart_rate  # 심박 주기 (초)
    
    # 심전도 신호 생성 (PQRST 파형 시뮬레이션)
    ecg = np.zeros_like(t)
    
    # 각 심박 주기마다 PQRST 파형 생성
    for i in range(int(duration / heart_period) + 1):
        cycle_start = i * heart_period
        cycle_t = t - cycle_start
        
        # QRS 복합체 (가장 큰 파형)
        qrs_mask = (cycle_t >= 0) & (cycle_t < 0.1)
        ecg[qrs_mask] += 1.5 * np.exp(-((cycle_t[qrs_mask] - 0.05) / 0.02) ** 2)
        
        # P 파
        p_mask = (cycle_t >= -0.2) & (cycle_t < 0)
        ecg[p_mask] += 0.2 * np.exp(-((cycle_t[p_mask] + 0.1) / 0.05) ** 2)
        
        # T 파
        t_mask = (cycle_t >= 0.15) & (cycle_t < 0.4)
        ecg[t_mask] += 0.3 * np.exp(-((cycle_t[t_mask] - 0.275) / 0.1) ** 2)
    
    # 노이즈 추가 (현실적인 심전도 신호를 위해)
    noise = np.random.normal(0, 0.05, len(ecg))
    ecg += noise
    
    return t, ecg

def plot_ecg(t, ecg, title="심전도 (ECG) 신호"):
    """
    심전도 신호를 그래프로 그립니다.
    
    Parameters:
    -----------
    t : numpy array
        시간 배열
    ecg : numpy array
        심전도 신호
    title : str
        그래프 제목
    """
    plt.figure(figsize=(14, 6))
    plt.plot(t, ecg, 'b-', linewidth=1.5, label='ECG 신호')
    plt.xlabel('시간 (초)', fontsize=12)
    plt.ylabel('전압 (mV)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # R 피크 찾기 및 표시
    peaks, properties = find_peaks(ecg, height=np.max(ecg) * 0.5, distance=int(0.5 * len(ecg) / (len(t) / t[-1])))
    if len(peaks) > 0:
        plt.plot(t[peaks], ecg[peaks], 'ro', markersize=8, label=f'R 피크 ({len(peaks)}개)')
        plt.legend(fontsize=10)
        
        # 심박수 계산
        if len(peaks) > 1:
            avg_rr = np.mean(np.diff(t[peaks]))
            heart_rate = 60.0 / avg_rr
            plt.text(0.02, 0.95, f'평균 심박수: {heart_rate:.1f} BPM', 
                    transform=plt.gca().transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def main():
    """메인 함수"""
    print("심전도 데이터 생성 중...")
    
    # 심전도 신호 생성 (10초간, 250Hz 샘플링)
    t, ecg = generate_ecg_signal(duration=10, sampling_rate=250)
    
    print(f"생성된 데이터 포인트 수: {len(ecg)}")
    print(f"시간 범위: 0 ~ {t[-1]:.1f} 초")
    print(f"신호 범위: {np.min(ecg):.3f} ~ {np.max(ecg):.3f} mV")
    
    # 그래프 그리기
    plot_ecg(t, ecg)
    
    # 통계 정보 출력
    peaks, _ = find_peaks(ecg, height=np.max(ecg) * 0.5, distance=int(0.5 * len(ecg) / (len(t) / t[-1])))
    if len(peaks) > 1:
        rr_intervals = np.diff(t[peaks])
        avg_rr = np.mean(rr_intervals)
        heart_rate = 60.0 / avg_rr
        print(f"\n심박 분석:")
        print(f"  R 피크 개수: {len(peaks)}")
        print(f"  평균 R-R 간격: {avg_rr:.3f} 초")
        print(f"  평균 심박수: {heart_rate:.1f} BPM")

if __name__ == "__main__":
    main()

