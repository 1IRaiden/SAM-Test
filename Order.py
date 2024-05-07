from abc import ABC, abstractmethod
from dataclasses import dataclass
import dacite
import numpy as np
import enum
import cv2



class AbstractShape(ABC):

    @dataclass
    class Settings:
        is_Perspective: bool = False
        is_active: bool = True
        color: tuple = (255, 255, 255)

    def __init__(self):
        self.setting: AbstractShape.Settings = self.Settings()

    @abstractmethod
    def get_cv(self):
        pass


class Circle(AbstractShape):

    @dataclass
    class Settings(AbstractShape.Settings):
        R: float = 0.4

    def __init__(self,  config: dict):
        super().__init__()
        self.setting: Circle.Settings = dacite.from_dict(Circle.Settings, config)

    def get_cv(self):
        pass


class Rectangle(AbstractShape):

    @dataclass
    class RectangleSettings(AbstractShape.Settings):
        dx: int = 20
        dy: int = 4
        dl: int = 40

    def __init__(self,  config: dict):
        super().__init__()
        self.setting: Rectangle.RectangleSettings = dacite.from_dict(Rectangle.RectangleSettings, config)
        self.dx_h = self.setting.dx

    def get_cv(self):
        pass

    def count_dx_h(self, value, max_heath):
        distance = (value/max_heath)*self.setting.dl
        self.dx_h = int(0 - self.setting.dx + distance)
class Any(AbstractShape):
    @dataclass
    class AnySettings(AbstractShape.Settings):
        is_active: bool = True
        dx: int = 80
        dy: int = 80

        dx_real: int = 1
        dy_real: int = 1

    def __init__(self, path, mask: str, config: dict):
        super().__init__()
        self.settings: Any.AnySettings = dacite.from_dict(Any.AnySettings, config)
        self.img = cv2.imread(path)
        self.mask = cv2.imread(mask, 0)
        if path.endswith(".png"):
            self.multiple_255()

        self.resize_img()

    def get_cv(self):
        return self.img

    def resize_img(self):
        self.img = cv2.resize(self.img, (80, 80))
        self.mask = cv2.resize(self.mask, (80, 80))
        print(self.mask.shape)

    def multiple_255(self):
        self.mask = (self.mask * 255).astype(np.uint8)


    # Здесь появится новый код
    # Я следаю попытку хакомминтится сонв
    # Думаем о возникновении новых строк
    # code
    # code
    # code
    # code






