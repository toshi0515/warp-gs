from pathlib import Path

import moderngl
import numpy as np

from warpgs.ply import load_ply
from warpgs.viewer_base import GSViewerBase


class GSViewer_PointCloud(GSViewerBase):
    title = "pointcloud viewer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 430 core
                in vec3 aPos; // ワールド座標系
                in vec3 aColor;
                uniform mat4 view;
                uniform mat4 projection;
                
                out vec3 localPos;
                out vec3 localColor;
                
                void main() {
                    gl_PointSize = 3;
                    gl_Position = projection * view * vec4(aPos, 1.0); // aPosはワールド座標系なのでmodel行列不要
                    localPos = aPos;
                    localColor = aColor; 
                }
            """,
            fragment_shader="""
                #version 430 core
                in vec3 localPos;
                in vec3 localColor;
                out vec4 FragColor;
                void main() {
                    FragColor = vec4(localColor, 1.0);
                }
            """,
        )

        # plyファイルを読み込む
        ply_path = Path(__file__).resolve().parents[1] / "assets" / "models" / "final.ply"
        vertics_dict = load_ply(ply_path)

        pos = vertics_dict["means"].numpy()
        color = vertics_dict["colors"].numpy()

        vertices = np.hstack([pos, color])  # (x, y, z, r, g, b)
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, "3f 3f", "aPos", "aColor")])

    def on_render(self, time, frame_time):
        self.ctx.clear(0.2, 0.3, 0.3, 1.0, depth=1.0)

        view, projection = super().move_camera(frame_time)
        self.prog["view"].write(view.to_bytes())
        self.prog["projection"].write(projection.to_bytes())

        self.vao.render(mode=self.ctx.POINTS)


if __name__ == "__main__":
    GSViewer_PointCloud.run()
