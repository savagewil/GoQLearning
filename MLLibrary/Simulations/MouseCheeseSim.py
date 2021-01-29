from typing import Tuple

import numpy

from MLLibrary.Models.Model import Model
from MLLibrary.Simulations.Simulation import Simulation


class MouseCheeseSim(Simulation):
    def __init__(self, size=9, cheese_limit=3):
        super().__init__()
        self.field = numpy.zeros((size, size))
        self.full_field = numpy.tile(self.field,(2,2))
        self.shifted_field = numpy.zeros((size, size))
        self.player_pos = numpy.array([0,0])
        self.size = size
        self.running = True
        self.cheese_limit = cheese_limit

    def run(self, model: Model) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float]:
        self.running = True
        while self.running:
            sim_state = self.state()
            self.display()
            sim_input = model.predict(sim_state)
            self.handle(sim_input)

    def handle(self, in_array:numpy.ndarray):
        move = in_array[:2]
        self.running = numpy.round(in_array[2]) == 0
        self.player_pos = numpy.mod(self.player_pos + numpy.round(move), self.size)

    def state(self) -> numpy.ndarray:
        return self.display_field.flatten()

    def update(self):
        if self.field[self.player_pos[0], self.player_pos[1]] == 1:
            self.score += 1
            self.field[self.player_pos[0], self.player_pos[1]] = 0
        while numpy.sum(self.field) < self.cheese_limit:
            row = numpy.random.randint(0, self.size - 1)
            col = numpy.random.randint(0, self.size - 1)
            while (row == self.player_pos[0] and col == self.player_pos[1]) and (self.field[row, col] == 1):
                row = numpy.random.randint(0, self.size - 1)
                col = numpy.random.randint(0, self.size - 1)
            self.field[row, col] = 1

        self.full_field = numpy.tile(self.field,(3,3))

        row_min = int(numpy.ceil(self.size/2) + self.player_pos[0])
        row_max = int(row_min + self.size)
        col_min = int(numpy.ceil(self.size/2) + self.player_pos[1])
        col_max = int(col_min + self.size)

        self.shifted_field = self.full_field[row_min:row_max, col_min:col_max]

    def display(self):
        self.update()
        row = self.size + self.player_pos[0]
        y = self.size + self.player_pos[1]
        self.display_field[row, y] = 2
        display_field =
        text = numpy.unicode(
            "\n".join(["".join([("⬟" if val == 2 else ("⬜" if val == 0 else "⬛")) for val in row]) for row in rotated_field])
        )
        print(text)
        print("Score %d"%(self.score))



    def left(self):
        self.handle(numpy.array([0, -1, 0]))

    def right(self):
        self.handle(numpy.array([0, 1, 0]))

    def up(self):
        self.handle(numpy.array([-1, 0, 0]))

    def down(self):
        self.handle(numpy.array([1, 0, 0]))

if __name__ == '__main__':
    MS = MouseCheeseSim()
    MS.display()
    MS.left()
    MS.display()
    MS.up()
    MS.display()
    MS.right()
    MS.display()
    MS.down()
    MS.display()
