"""
삼변측량법 시각화 프로그램
사용자가 원 안에 점을 찍으면 A, B, C를 기준으로 삼변측량법을 통해 위치를 찾는 모습을 보여줍니다.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # Windows에서 한글 폰트 설정
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


class TriangulationVisualization:
    def __init__(self):
        # 좌표계 설정
        self.x_min, self.x_max = -150, 150
        self.y_min, self.y_max = -150, 150
        
        # 원 P: 중심 원점, 반지름 30
        self.circle_p_radius = 30
        
        # 점 A, B, C: 중심 원점, 반지름 80인 원 위를 돌고 있음
        self.reference_radius = 80
        self.angle_a = 0  # A의 초기 각도
        self.angle_b = 120 * np.pi / 180  # B의 초기 각도 (120도)
        self.angle_c = 240 * np.pi / 180  # C의 초기 각도 (240도)
        self.rotation_speed = 0.02  # 회전 속도
        
        # 사용자가 찍은 점 S
        self.point_s = None
        
        # 애니메이션 제어
        self.animation = None
        self.is_animating = True
        
        # 그림 설정
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X 좌표', fontsize=12)
        self.ax.set_ylabel('Y 좌표', fontsize=12)
        self.ax.set_title('삼변측량법 시각화 - 원 안을 클릭하여 점 S를 찍으세요', fontsize=14, fontweight='bold')
        
        # 클릭 이벤트 연결
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # 리셋 버튼
        ax_reset = plt.axes([0.85, 0.02, 0.1, 0.04])
        self.btn_reset = Button(ax_reset, '리셋')
        self.btn_reset.on_clicked(self.reset)
        
        # 초기 그리기
        self.draw()
        
    def get_reference_points(self):
        """현재 각도에 따른 기준점 A, B, C의 좌표를 반환"""
        a_x = self.reference_radius * np.cos(self.angle_a)
        a_y = self.reference_radius * np.sin(self.angle_a)
        b_x = self.reference_radius * np.cos(self.angle_b)
        b_y = self.reference_radius * np.sin(self.angle_b)
        c_x = self.reference_radius * np.cos(self.angle_c)
        c_y = self.reference_radius * np.sin(self.angle_c)
        
        return np.array([a_x, a_y]), np.array([b_x, b_y]), np.array([c_x, c_y])
    
    def calculate_distances(self, point_s):
        """점 S에서 각 기준점까지의 거리를 계산"""
        a, b, c = self.get_reference_points()
        dist_a = np.linalg.norm(point_s - a)
        dist_b = np.linalg.norm(point_s - b)
        dist_c = np.linalg.norm(point_s - c)
        return dist_a, dist_b, dist_c
    
    def triangulate(self, point_a, point_b, point_c, dist_a, dist_b, dist_c):
        """
        삼변측량법을 사용하여 위치를 계산
        A, B, C를 중심으로 한 세 원이 모두 만나는 점을 찾습니다.
        방법: 각 두 원의 교점 중 세 번째 원에 가장 가까운 점을 선택하고,
        세 점의 평균을 구합니다.
        """
        def circle_intersection(c1, r1, c2, r2):
            """두 원의 교점을 계산"""
            d = np.linalg.norm(c2 - c1)
            
            # 원이 겹치지 않는 경우
            if d > r1 + r2 or d < abs(r1 - r2):
                return None, None
            
            # 중점
            a = (r1**2 - r2**2 + d**2) / (2 * d)
            h_sq = r1**2 - a**2
            
            # 허수 해 방지
            if h_sq < 0:
                return None, None
            
            h = np.sqrt(h_sq)
            
            # 중점 좌표
            p2 = c1 + a * (c2 - c1) / d
            
            # 교점 좌표
            dx = (c2[0] - c1[0]) / d
            dy = (c2[1] - c1[1]) / d
            
            x3 = p2[0] + h * dy
            y3 = p2[1] - h * dx
            x4 = p2[0] - h * dy
            y4 = p2[1] + h * dx
            
            return np.array([x3, y3]), np.array([x4, y4])
        
        # 1. A-B 원의 교점 중에서 C 원에 가장 가까운 점 선택
        p1_ab, p2_ab = circle_intersection(point_a, dist_a, point_b, dist_b)
        point_from_ab = None
        if p1_ab is not None and p2_ab is not None:
            # C 원까지의 거리 오차 계산
            error_p1 = abs(np.linalg.norm(p1_ab - point_c) - dist_c)
            error_p2 = abs(np.linalg.norm(p2_ab - point_c) - dist_c)
            point_from_ab = p1_ab if error_p1 < error_p2 else p2_ab
        
        # 2. A-C 원의 교점 중에서 B 원에 가장 가까운 점 선택
        p1_ac, p2_ac = circle_intersection(point_a, dist_a, point_c, dist_c)
        point_from_ac = None
        if p1_ac is not None and p2_ac is not None:
            # B 원까지의 거리 오차 계산
            error_p1 = abs(np.linalg.norm(p1_ac - point_b) - dist_b)
            error_p2 = abs(np.linalg.norm(p2_ac - point_b) - dist_b)
            point_from_ac = p1_ac if error_p1 < error_p2 else p2_ac
        
        # 3. B-C 원의 교점 중에서 A 원에 가장 가까운 점 선택
        p1_bc, p2_bc = circle_intersection(point_b, dist_b, point_c, dist_c)
        point_from_bc = None
        if p1_bc is not None and p2_bc is not None:
            # A 원까지의 거리 오차 계산
            error_p1 = abs(np.linalg.norm(p1_bc - point_a) - dist_a)
            error_p2 = abs(np.linalg.norm(p2_bc - point_a) - dist_a)
            point_from_bc = p1_bc if error_p1 < error_p2 else p2_bc
        
        # 4. 세 점의 평균을 구하여 추정 위치 결정
        valid_points = []
        if point_from_ab is not None:
            valid_points.append(point_from_ab)
        if point_from_ac is not None:
            valid_points.append(point_from_ac)
        if point_from_bc is not None:
            valid_points.append(point_from_bc)
        
        if len(valid_points) == 0:
            # 모든 방법이 실패한 경우, 기준점들의 중심을 반환
            return (point_a + point_b + point_c) / 3
        
        # 세 점의 평균 (또는 가중 평균)
        estimated_point = np.mean(valid_points, axis=0)
        
        return estimated_point
    
    def on_click(self, event):
        """마우스 클릭 이벤트 처리"""
        if event.inaxes != self.ax:
            return
        
        # 클릭한 위치
        click_x, click_y = event.xdata, event.ydata
        
        # 원 P 안에 있는지 확인
        distance_from_origin = np.sqrt(click_x**2 + click_y**2)
        
        if distance_from_origin <= self.circle_p_radius:
            self.point_s = np.array([click_x, click_y])
            # 점을 찍으면 애니메이션 멈추기
            if self.animation is not None and self.is_animating:
                self.animation.event_source.stop()
                self.is_animating = False
            self.draw()
        else:
            print(f"원 P 안에 점을 찍어주세요. (현재 거리: {distance_from_origin:.2f}, 최대: {self.circle_p_radius})")
    
    def reset(self, event):
        """리셋 버튼 클릭 시"""
        self.point_s = None
        # 리셋하면 애니메이션 다시 시작
        if self.animation is not None and not self.is_animating:
            self.animation.event_source.start()
            self.is_animating = True
        self.draw()
    
    def draw(self):
        """전체 그래프를 그리기"""
        self.ax.clear()
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X 좌표', fontsize=12)
        self.ax.set_ylabel('Y 좌표', fontsize=12)
        
        # 원점 표시
        self.ax.plot(0, 0, 'ko', markersize=8, label='원점')
        
        # 원 P 그리기 (중심 원점, 반지름 30)
        circle_p = patches.Circle((0, 0), self.circle_p_radius, 
                                  fill=False, edgecolor='blue', linewidth=2, linestyle='--')
        self.ax.add_patch(circle_p)
        self.ax.text(0, self.circle_p_radius + 5, '원 P (반지름 30)', 
                    ha='center', fontsize=10, color='blue')
        
        # 기준점 A, B, C 가져오기
        point_a, point_b, point_c = self.get_reference_points()
        
        # 기준점 A, B, C 표시
        self.ax.plot(point_a[0], point_a[1], 'ro', markersize=10, label='기준점 A')
        self.ax.plot(point_b[0], point_b[1], 'go', markersize=10, label='기준점 B')
        self.ax.plot(point_c[0], point_c[1], 'mo', markersize=10, label='기준점 C')
        
        # 기준점 라벨
        self.ax.text(point_a[0] + 3, point_a[1] + 3, 'A', fontsize=12, fontweight='bold', color='red')
        self.ax.text(point_b[0] + 3, point_b[1] + 3, 'B', fontsize=12, fontweight='bold', color='green')
        self.ax.text(point_c[0] + 3, point_c[1] + 3, 'C', fontsize=12, fontweight='bold', color='magenta')
        
        # 기준점들이 돌고 있는 원 그리기 (반지름 80)
        reference_circle = patches.Circle((0, 0), self.reference_radius, 
                                         fill=False, edgecolor='gray', linewidth=1, linestyle=':')
        self.ax.add_patch(reference_circle)
        
        # 점 S가 찍혔을 때
        if self.point_s is not None:
            # 점 S 표시
            self.ax.plot(self.point_s[0], self.point_s[1], 'bs', markersize=12, label='점 S')
            self.ax.text(self.point_s[0] + 3, self.point_s[1] + 3, 'S', 
                        fontsize=12, fontweight='bold', color='blue')
            
            # 각 기준점에서 점 S까지의 거리 계산
            dist_a, dist_b, dist_c = self.calculate_distances(self.point_s)
            
            # 각 기준점을 중심으로 점 S를 지나는 원 그리기
            circle_a = patches.Circle((point_a[0], point_a[1]), dist_a, 
                                     fill=False, edgecolor='red', linewidth=2, alpha=0.7, linestyle='-')
            circle_b = patches.Circle((point_b[0], point_b[1]), dist_b, 
                                     fill=False, edgecolor='green', linewidth=2, alpha=0.7, linestyle='-')
            circle_c = patches.Circle((point_c[0], point_c[1]), dist_c, 
                                     fill=False, edgecolor='magenta', linewidth=2, alpha=0.7, linestyle='-')
            
            self.ax.add_patch(circle_a)
            self.ax.add_patch(circle_b)
            self.ax.add_patch(circle_c)
            
            # 기준점에서 점 S까지의 선 그리기
            self.ax.plot([point_a[0], self.point_s[0]], [point_a[1], self.point_s[1]], 
                        'r--', linewidth=1, alpha=0.5)
            self.ax.plot([point_b[0], self.point_s[0]], [point_b[1], self.point_s[1]], 
                        'g--', linewidth=1, alpha=0.5)
            self.ax.plot([point_c[0], self.point_s[0]], [point_c[1], self.point_s[1]], 
                        'm--', linewidth=1, alpha=0.5)
            
            # 거리 정보 표시
            self.ax.text(point_a[0] + 5, point_a[1] - 5, f'd={dist_a:.1f}', 
                        fontsize=9, color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            self.ax.text(point_b[0] + 5, point_b[1] - 5, f'd={dist_b:.1f}', 
                        fontsize=9, color='green', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            self.ax.text(point_c[0] + 5, point_c[1] - 5, f'd={dist_c:.1f}', 
                        fontsize=9, color='magenta', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # 삼변측량법으로 위치 추정
            estimated_point = self.triangulate(point_a, point_b, point_c, dist_a, dist_b, dist_c)
            
            if estimated_point is not None:
                # 추정된 위치 표시
                self.ax.plot(estimated_point[0], estimated_point[1], 'y*', 
                           markersize=20, label='추정 위치', zorder=10)
                self.ax.text(estimated_point[0] + 3, estimated_point[1] + 3, '추정', 
                           fontsize=11, fontweight='bold', color='orange',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                
                # 실제 위치와 추정 위치의 오차 계산
                error = np.linalg.norm(estimated_point - self.point_s)
                self.ax.text(0.02, 0.98, 
                           f'실제 위치: ({self.point_s[0]:.2f}, {self.point_s[1]:.2f})\n'
                           f'추정 위치: ({estimated_point[0]:.2f}, {estimated_point[1]:.2f})\n'
                           f'오차: {error:.2f}',
                           transform=self.ax.transAxes, fontsize=10,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # 제목 업데이트
            self.ax.set_title('삼변측량법 시각화 - 점 S가 설정되었습니다', 
                            fontsize=14, fontweight='bold')
        else:
            self.ax.set_title('삼변측량법 시각화 - 원 안을 클릭하여 점 S를 찍으세요', 
                            fontsize=14, fontweight='bold')
        
        self.ax.legend(loc='upper right', fontsize=9)
        self.fig.canvas.draw()
    
    def animate(self):
        """애니메이션으로 기준점들이 회전하도록"""
        import matplotlib.animation as animation
        
        def update(frame):
            # 점이 찍혀있으면 애니메이션 중지
            if self.point_s is not None:
                return []
            
            self.angle_a += self.rotation_speed
            self.angle_b += self.rotation_speed
            self.angle_c += self.rotation_speed
            self.draw()
            return []
        
        ani = animation.FuncAnimation(self.fig, update, interval=50, blit=False, repeat=True)
        return ani
    
    def show(self):
        """그래프 표시"""
        plt.tight_layout()
        plt.show()


def main():
    """메인 함수"""
    print("삼변측량법 시각화 프로그램을 시작합니다.")
    print("=" * 50)
    print("사용 방법:")
    print("1. 파란색 점선 원(원 P) 안을 클릭하여 점 S를 찍으세요.")
    print("2. 빨간색, 초록색, 자주색 원이 각각 기준점 A, B, C를 중심으로 그려집니다.")
    print("3. 노란색 별표로 추정된 위치가 표시됩니다.")
    print("4. '리셋' 버튼을 눌러 점 S를 초기화할 수 있습니다.")
    print("=" * 50)
    
    viz = TriangulationVisualization()
    
    # 애니메이션 시작
    viz.animation = viz.animate()
    
    viz.show()


if __name__ == "__main__":
    main()

