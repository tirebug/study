"""
사다리꼴 공식을 사용한 정적분 시각화 프로그램
구간 [a, b]를 n개의 작은 사다리꼴로 나누어 정적분 ∫[a,b] f(x) dx의 값을 근사합니다.
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Windows에서 GUI 백엔드 명시적 설정
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy import integrate
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # Windows에서 한글 폰트 설정
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
plt.ioff()  # Interactive mode 비활성화 (blocking mode)


class TrapezoidalIntegration:
    def __init__(self, func_dict, n_min=1, n_max=100, n_init=5):
        """
        Parameters:
        -----------
        func_dict : dict
            함수 이름을 키로, (함수, a, b, 함수식 문자열) 튜플을 값으로 하는 딕셔너리
            예: {'x²': (lambda x: x**2, 0, 2, 'x²')}
        n_min : int
            사다리꼴 개수의 최소값
        n_max : int
            사다리꼴 개수의 최대값
        n_init : int
            초기 사다리꼴 개수
        """
        self.func_dict = func_dict
        self.func_names = list(func_dict.keys())
        self.current_func_name = self.func_names[0]
        self.func, self.a, self.b, self.func_label = func_dict[self.current_func_name]
        
        self.n_min = n_min
        self.n_max = n_max
        self.n_init = n_init
        self.n = n_init
        
        # 고정된 좌표평면 범위 설정 (구간이 바뀌어도 유지)
        self.fixed_x_min = -10
        self.fixed_x_max = 10
        self.fixed_y_min = -10
        self.fixed_y_max = 10
        
        # 그림 설정
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        # 좌표평면을 가운데 최대 크기로 배치 (왼쪽 라디오 버튼과 오른쪽 슬라이더 공간 확보)
        plt.subplots_adjust(bottom=0.1, left=0.14, right=0.70, top=0.95)
        
        # 함수 선택 라디오 버튼 (왼쪽에 배치)
        ax_radio = plt.axes([0.01, 0.1, 0.11, 0.25])
        self.radio = RadioButtons(ax_radio, self.func_names)
        self.radio.on_clicked(self.change_func)
        # 라디오 버튼 텍스트 크기 조정
        for label in self.radio.labels:
            label.set_fontsize(9)
        
        # 리셋 버튼 (라디오 버튼 위에 배치)
        ax_reset = plt.axes([0.01, 0.36, 0.11, 0.04])
        self.button_reset = Button(ax_reset, '리셋')
        self.button_reset.on_clicked(self.reset)
        
        # 오른쪽에 세로 슬라이더 배치 (길이를 늘리고 간격 조정)
        # 구간 하한 (a) 슬라이더 - 세로
        ax_slider_a = plt.axes([0.72, 0.15, 0.02, 0.7])
        self.slider_a = Slider(
            ax_slider_a, '구간\n하한 (a)', 
            -10, 10, 
            valinit=self.a, 
            valstep=0.1,
            orientation='vertical'
        )
        self.slider_a.on_changed(self.update_interval)
        # 슬라이더 레이블 폰트 설정 (음수 표시 문제 해결)
        self.slider_a.label.set_fontsize(9)
        self.slider_a.valtext.set_fontsize(8)
        self.slider_a.valtext.set_fontfamily('DejaVu Sans')  # 음수 표시를 위한 폰트 설정
        
        # 구간 상한 (b) 슬라이더 - 세로
        ax_slider_b = plt.axes([0.76, 0.15, 0.02, 0.7])
        self.slider_b = Slider(
            ax_slider_b, '구간\n상한 (b)', 
            -10, 10, 
            valinit=self.b, 
            valstep=0.1,
            orientation='vertical'
        )
        self.slider_b.on_changed(self.update_interval)
        # 슬라이더 레이블 폰트 설정
        self.slider_b.label.set_fontsize(9)
        self.slider_b.valtext.set_fontsize(8)
        self.slider_b.valtext.set_fontfamily('DejaVu Sans')  # 음수 표시를 위한 폰트 설정
        
        # 사다리꼴 개수 슬라이더 - 세로
        ax_slider = plt.axes([0.80, 0.15, 0.02, 0.7])
        self.slider = Slider(
            ax_slider, '사다리꼴\n개수 (n)', 
            n_min, n_max, 
            valinit=n_init, 
            valstep=1,
            orientation='vertical'
        )
        self.slider.on_changed(self.update)
        # 슬라이더 레이블 폰트 설정
        self.slider.label.set_fontsize(9)
        self.slider.valtext.set_fontsize(8)
        
        # 초기 플롯 (슬라이더 생성 후)
        self.update_plot()
        
        plt.show(block=True)  # blocking mode로 창 유지
    
    def change_func(self, label):
        """함수가 변경될 때 호출됩니다."""
        self.current_func_name = label
        self.func, self.a, self.b, self.func_label = self.func_dict[label]
        # 슬라이더 값 업데이트
        self.slider_a.set_val(self.a)
        self.slider_b.set_val(self.b)
        self.update_plot()
    
    def update_interval(self, val):
        """구간이 변경될 때 호출됩니다."""
        self.a = self.slider_a.val
        self.b = self.slider_b.val
        # a < b가 되도록 보장
        if self.a >= self.b:
            if val == self.slider_a.val:
                self.b = self.a + 0.1
                self.slider_b.set_val(self.b)
            else:
                self.a = self.b - 0.1
                self.slider_a.set_val(self.a)
        self.update_plot()
    
    def format_scientific(self, value):
        """
        과학적 표기법을 10의 제곱 형식으로 변환합니다.
        예: 1.23e-04 → 1.23 × 10⁻⁴
        """
        # 위첨자 문자 매핑
        superscript_map = str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")
        
        # 과학적 표기법으로 변환
        sci_str = f'{value:.2e}'
        
        # mantissa와 exponent 분리
        if 'e' in sci_str:
            mantissa, exponent = sci_str.split('e')
            # exponent를 위첨자로 변환
            exp_superscript = exponent.translate(superscript_map)
            return f'{mantissa} × 10{exp_superscript}'
        else:
            return sci_str
    
    def trapezoidal_rule(self, n):
        """
        사다리꼴 공식을 사용하여 정적분을 근사합니다.
        
        Parameters:
        -----------
        n : int
            사다리꼴의 개수
            
        Returns:
        --------
        integral_value : float
            근사된 적분값
        x_points : array
            x 좌표 점들
        y_points : array
            y 좌표 점들
        """
        # 구간을 n개로 나누기
        x_points = np.linspace(self.a, self.b, n + 1)
        y_points = self.func(x_points)
        
        # 사다리꼴 공식: h/2 * [f(x0) + 2*f(x1) + 2*f(x2) + ... + 2*f(x_{n-1}) + f(xn)]
        h = (self.b - self.a) / n
        integral_value = (h / 2) * (y_points[0] + 2 * np.sum(y_points[1:-1]) + y_points[-1])
        
        return integral_value, x_points, y_points
    
    def update_plot(self):
        """그림을 업데이트합니다."""
        self.ax.clear()
        
        # 정확한 함수 곡선 그리기
        x_fine = np.linspace(self.a, self.b, 1000)
        y_fine = self.func(x_fine)
        self.ax.plot(x_fine, y_fine, 'b-', linewidth=2, label=f'f(x)')
        
        # 사다리꼴 근사 계산
        integral_value, x_points, y_points = self.trapezoidal_rule(self.n)
        
        # 사다리꼴 그리기
        for i in range(self.n):
            x_left = x_points[i]
            x_right = x_points[i + 1]
            y_left = y_points[i]
            y_right = y_points[i + 1]
            
            # 사다리꼴의 네 꼭짓점
            trapezoid_x = [x_left, x_right, x_right, x_left, x_left]
            trapezoid_y = [0, 0, y_right, y_left, 0]
            
            # 사다리꼴 채우기
            self.ax.fill(trapezoid_x, trapezoid_y, alpha=0.3, color='green', edgecolor='darkgreen', linewidth=1.5)
            
            # 사다리꼴의 윗변 그리기
            self.ax.plot([x_left, x_right], [y_left, y_right], 'g-', linewidth=2)
        
        # 함수 위의 점들 표시
        self.ax.plot(x_points, y_points, 'ro', markersize=6, label='분할점')
        
        # y축 범위 동적 조정 (함수 값이 모두 양수이면 양수 부분만 표시)
        y_min_value = np.min(y_fine)
        y_max_value = np.max(y_fine)
        
        # x축은 고정된 범위 유지
        self.ax.set_xlim(self.fixed_x_min, self.fixed_x_max)
        
        # y축 범위 설정
        if y_min_value >= 0:
            # 함수 값이 모두 양수이면 0부터 시작
            y_padding = (y_max_value - 0) * 0.1
            self.ax.set_ylim(0, y_max_value + y_padding)
        else:
            # 음수 값이 있으면 기존처럼 표시
            y_padding = (y_max_value - y_min_value) * 0.1
            self.ax.set_ylim(y_min_value - y_padding, y_max_value + y_padding)
        
        # x축과 y축 비율을 1:1로 설정 (격자가 정사각형으로 보이도록)
        # adjustable='box'를 사용하여 그래프 영역이 조정되도록 함
        self.ax.set_aspect('equal', adjustable='box')
        
        self.ax.grid(True, alpha=0.3)
        self.ax.axhline(y=0, color='k', linewidth=0.5)
        self.ax.axvline(x=0, color='k', linewidth=0.5)
        
        # 정확한 적분값 계산
        exact_value, _ = integrate.quad(self.func, self.a, self.b)
        
        # 제목과 레이블
        self.ax.set_xlabel('x', fontsize=12)
        self.ax.set_ylabel('f(x)', fontsize=12)
        self.ax.set_title(
            f'사다리꼴 공식으로 정적분 근사: f(x) = {self.func_label}, 구간: [{self.a:.2f}, {self.b:.2f}]',
            fontsize=13, fontweight='bold', pad=20
        )
        
        # 적분값을 오른쪽 상단에 텍스트로 표시
        error = abs(integral_value - exact_value)
        # 오차를 적절한 형식으로 표시
        if error < 0.001:
            # 과학적 표기법을 10의 제곱 형식으로 변환
            error_str = self.format_scientific(error)
        else:
            error_str = f'{error:.6f}'
        
        info_text = (
            f'사다리꼴 개수: n = {self.n}\n'
            f'근사값: ∫f(x)dx ≈ {integral_value:.6f}\n'
            f'정확한 값: ∫f(x)dx = {exact_value:.6f}\n'
            f'오차: {error_str}'
        )
        self.ax.text(0.98, 0.95, info_text, 
                    transform=self.ax.transAxes,
                    fontsize=10,
                    fontfamily='DejaVu Sans',  # 등호가 제대로 표시되는 폰트 사용
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.ax.legend(loc='upper left')
        
        self.fig.canvas.draw()
    
    def update(self, val):
        """슬라이더 값이 변경될 때 호출됩니다."""
        self.n = int(self.slider.val)
        self.update_plot()
    
    def reset(self, event):
        """리셋 버튼을 클릭했을 때 호출됩니다."""
        # 함수의 초기 구간으로 리셋
        _, init_a, init_b, _ = self.func_dict[self.current_func_name]
        self.slider_a.set_val(init_a)
        self.slider_b.set_val(init_b)
        self.slider.set_val(self.n_init)
        self.a = init_a
        self.b = init_b
        self.n = self.n_init
        self.update_plot()


# 예제 함수들
def example_func1(x):
    """예제 함수 1: x^2"""
    return x ** 2

def example_func2(x):
    """예제 함수 2: sin(x)"""
    return np.sin(x)

def example_func3(x):
    """예제 함수 3: e^x"""
    return np.exp(x)

def example_func4(x):
    """예제 함수 4: x^3 - 2*x + 1"""
    return x ** 3 - 2 * x + 1

def example_func5(x):
    """예제 함수 5: x^4 - 2*x^2 + 1 (사차함수)"""
    return x ** 4 - 2 * x ** 2 + 1


if __name__ == "__main__":
    # 사용 예시
    # 함수 딕셔너리: {함수이름: (함수, a, b, 함수식_문자열)}
    func_dict = {
        'x²': (example_func1, 0, 2, 'x²'),
        'sin(x)': (example_func2, 0, np.pi, 'sin(x)'),
        'e^x': (example_func3, 0, 1, 'e^x'),
        'x³-2x+1': (example_func4, -1, 2, 'x³ - 2x + 1'),
        'x⁴-2x²+1': (example_func5, -2, 2, 'x⁴ - 2x² + 1')
    }
    
    print("사다리꼴 공식을 사용한 정적분 시각화 프로그램")
    print("=" * 50)
    print("왼쪽 라디오 버튼으로 함수를 선택할 수 있습니다.")
    print("슬라이더를 조절하여 사다리꼴 개수를 변경할 수 있습니다.")
    print("사다리꼴 개수가 많을수록 더 정확한 근사값을 얻을 수 있습니다.")
    print("=" * 50)
    
    viz = TrapezoidalIntegration(
        func_dict=func_dict,
        n_min=1,
        n_max=50,
        n_init=5
    )

