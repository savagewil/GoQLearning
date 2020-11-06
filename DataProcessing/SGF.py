BLACK_PLAYER
WHITE_PLAYER
EVENT
ROUND
MOVE_WHITE
MOVE_BLACK
PREMOVE_WHITE
PREMOVE_BLACK
SIZE

class SGF():
    def __init__(self):
        self.Black_Player = None
        self.White_Player = None
        self.Event = None
        self.Round = None
        self.Moves = []
        self.PreMoves = []
        self.Boards = {}
        self.Size = [19, 19]

    @staticmethod
    def FromFile(path):
        sgf_file = open(path, "r")
        return SGF.FromString(sgf_file)

    @staticmethod
    def FromString(string):
        sgf_data = SGF

        return sgf_data

    def __str__(self):
        return ""