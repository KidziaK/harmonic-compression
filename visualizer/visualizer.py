import numpy as np

from numpy.typing import NDArray
from OpenGL.GL import *
from OpenGL.GLU import *
from imgui.integrations.glfw import GlfwRenderer
from typing import Callable

import imgui
import glfw
import math
import numba
import cv2


class GLFWError(Exception):
    """Raised when GLFW encounters an error"""
    pass

class SphericalMapVisualizer:
    def __init__(self, heat_map: NDArray[np.uint8]) -> None:
        if not glfw.init():
            raise GLFWError("Failed to initialize glfw.")

        window_size = (800, 600)
        self.window = glfw.create_window(*window_size, "Sphere Heatmap", None, None)
        if not self.window:
            glfw.terminate()
            raise GLFWError("Failed to create a glfw window.")

        self.heat_map = heat_map

        glfw.make_context_current(self.window)
        
        # Initialize ImGui first
        imgui.create_context()
        self.impl = GlfwRenderer(self.window)
        
        # Now set callbacks
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        
        # Initialize mouse state
        self.last_x = window_size[0] / 2
        self.last_y = window_size[1] / 2
        self.first_mouse = True
        self.is_rotating = False
        
        # Camera parameters
        self.camera_distance = 8.0
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.sensitivity = 1.0
        self.zoom_sensitivity = 0.5

        self.sphere = gluNewQuadric()

        # Set up OpenGL state
        glEnable(GL_DEPTH_TEST)
        
        # Set up projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (window_size[0] / window_size[1]), 0.1, 50.0)
        
        self.heat_map_id = self.generate_texture()

    def mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.is_rotating = True
                self.first_mouse = True
            elif action == glfw.RELEASE:
                self.is_rotating = False

    def mouse_callback(self, window, xpos, ypos):
        if not self.is_rotating:
            return
            
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False
            return
        
        x_offset = (xpos - self.last_x) * self.sensitivity
        y_offset = (ypos - self.last_y) * self.sensitivity
        
        self.rotation_y += x_offset
        self.rotation_x += y_offset
        
        self.last_x = xpos
        self.last_y = ypos

    def scroll_callback(self, window, xoffset, yoffset):
        self.camera_distance -= yoffset * self.zoom_sensitivity
        self.camera_distance = np.clip(self.camera_distance, 2.0, 20.0)

    def generate_texture(self) -> int:
        width, height, *_ = self.heat_map.shape
        heat_map_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, heat_map_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, self.heat_map)
        glBindTexture(GL_TEXTURE_2D, 0)
        return heat_map_id

    def render_sphere(self) -> None:
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Move camera back
        glTranslatef(0, 0, -self.camera_distance)
        
        # Apply rotations
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        
        # Draw the sphere
        glColor3f(1, 1, 1)
        gluQuadricTexture(self.sphere, GL_TRUE)
        gluQuadricNormals(self.sphere, GLU_SMOOTH)
        
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.heat_map_id)
        gluSphere(self.sphere, 2.0, 36, 18)
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)

    def run(self) -> None:
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.impl.process_inputs()
            
            imgui.new_frame()
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            background_color = (0.3, 0.3, 0.3, 1.0)
            glClearColor(*background_color)
            
            self.render_sphere()
            
            # Debug window
            imgui.begin("Debug Info")
            imgui.text(f"Rotation X: {self.rotation_x:.2f}")
            imgui.text(f"Rotation Y: {self.rotation_y:.2f}")
            imgui.text(f"Is Rotating: {self.is_rotating}")
            imgui.text(f"Camera Distance: {self.camera_distance:.2f}")
            imgui.end()
            
            imgui.render()
            self.impl.render(imgui.get_draw_data())
            
            glfw.swap_buffers(self.window)

        self.impl.shutdown()
        glfw.terminate()

if __name__ == "__main__":
    app = SphericalMapVisualizer()
    app.run()