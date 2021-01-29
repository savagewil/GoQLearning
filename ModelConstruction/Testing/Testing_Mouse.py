from MLLibrary.Models.TextInputModel import TextInputModel
from MLLibrary.Simulations.MouseCheeseSim import MouseCheeseSim

MS = MouseCheeseSim()

response_table = {
    "a":[0, -1, 0],
    "d":[0, 1, 0],
    "w":[-1, 0, 0],
    "s":[1, 0, 0],
    "q":[0, 0, 1]
}
TM = TextInputModel(3, response_table)

MS.run(TM)