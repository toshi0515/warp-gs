import math

import glm
import moderngl_window as mglw


class GSViewerBase(mglw.WindowConfig):
    gl_version = (4, 3)
    window_size = (800, 600)
    aspect_ratio = 800 / 600

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # カメラの位置と向き
        self.cameraPos = glm.vec3(0.0, 0.0, 3.0)  # 最初は少し手前(Z=3.0)に配置
        self.cameraFront = glm.vec3(0.0, 0.0, -1.0)
        self.cameraUp = glm.vec3(0.0, 1.0, 0.0)

        # マウス視点移動用 (オイラー角)
        self.yaw = -90.0  # 最初から-Z方向(奥)を向くように-90度に設定
        self.pitch = 0.0  # 上下方向の角度
        self.fov = 45.0  # 視野角(ズーム用)

        # キーボード入力状態の保存用辞書
        self.keys = {self.wnd.keys.W: False, self.wnd.keys.A: False, self.wnd.keys.S: False, self.wnd.keys.D: False}

    def on_key_event(self, key, action, modifiers):
        # ESCキーを押したらウィンドウを閉じる
        if key == self.wnd.keys.ESCAPE and action == self.wnd.keys.ACTION_PRESS:
            self.wnd.close()

        # 押されたキーの状態を辞書に保存する
        if key in self.keys:
            if action == self.wnd.keys.ACTION_PRESS:
                self.keys[key] = True
            elif action == self.wnd.keys.ACTION_RELEASE:
                self.keys[key] = False

    def on_mouse_position_event(self, x, y, dx, dy):
        sensitivity = 0.1  # マウス感度
        self.yaw += dx * sensitivity
        self.pitch -= dy * sensitivity  # Y軸の動きを反転

        # ジンバルロック対策
        self.pitch = min(self.pitch, 89.0)  # 上方向の制限
        self.pitch = max(self.pitch, -89.0)  # 下方向の制限

        # オイラー角(yaw, pitch)から、向いている方向(ベクトル)を計算
        front = glm.vec3()
        front.x = math.cos(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        front.y = math.sin(glm.radians(self.pitch))
        front.z = math.sin(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        self.cameraFront = glm.normalize(front)

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.fov -= y_offset * 2.0
        self.fov = max(self.fov, 1.0)
        self.fov = min(self.fov, 45.0)

    def move_camera(self, frame_time):
        # frame_timeを掛けることで一定速度で移動する
        cameraSpeed = 2.5 * frame_time

        if self.keys[self.wnd.keys.W]:
            self.cameraPos += cameraSpeed * self.cameraFront  # 前進
        if self.keys[self.wnd.keys.S]:
            self.cameraPos -= cameraSpeed * self.cameraFront  # 後退

        # 向いている方向と上方向の外積で右方向を作る
        camera_right = glm.normalize(glm.cross(self.cameraFront, self.cameraUp))
        if self.keys[self.wnd.keys.A]:
            self.cameraPos -= camera_right * cameraSpeed  # 左
        if self.keys[self.wnd.keys.D]:
            self.cameraPos += camera_right * cameraSpeed  # 右

        # View行列: glm.lookAt(カメラの位置, 見つめる目標地点, 上方向)
        view = glm.lookAt(self.cameraPos, self.cameraPos + self.cameraFront, self.cameraUp)

        # Projection行列: fov (視野角) を反映させてズームを表現
        projection = glm.perspective(glm.radians(self.fov), self.aspect_ratio, 0.1, 100.0)

        return view, projection
