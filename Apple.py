class Apple:
    def __init__(self, num, x, y, wid, hei):
        self.num = num
        self.x = x
        self.y = y
        self.endx = x+wid
        self.endy = y+hei
    def cut(self):
        self.num = 0
  
if __name__ == '__main__':
    apple = Apple(0, 10, 10, 10, 10)
    print(apple.cut())