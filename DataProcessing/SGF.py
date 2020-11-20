EVENT = "EV"
ROUND = "RO"
SIZE = "SZ"
WHITE_PLAYER = "PW"
BLACK_PLAYER = "PB"
MOVE_WHITE = "W"
MOVE_BLACK = "B"
PREMOVE_WHITE = "AW"
PREMOVE_BLACK = "AB"


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
    def FromString(SGF_String:str):
        sgf_data = SGF()

        sgf_data.Event, index = getData(SGF_String, EVENT)

        sgf_data.Round, index = getData(SGF_String, ROUND)

        sgf_data.White_Player, index = getData(SGF_String, WHITE_PLAYER)

        sgf_data.Black_Player, index = getData(SGF_String, BLACK_PLAYER)

        # ===========================
        index = 0
        while True:
            text, index = getData(SGF_String, PREMOVE_BLACK, index)
            if index >= 0:
                break
            else:
                sgf_data.PreMoves.append(text)

        index = 0
        while True:
            text, index = getData(SGF_String, PREMOVE_WHITE, index)
            if index >= 0:
                break
            else:
                sgf_data.PreMoves.append(text)
        # ===========================
        color = 0
        index = 0
        while True:
            text, index = getData(SGF_String, PREMOVE_BLACK, index)
            if index >= 0:
                break
            else:
                sgf_data.PreMoves.append(text)
        return sgf_data

    def __str__(self):
        return ""


def getData(sgf_string, key, start_index=0):
    start = sgf_string.find(key + "[") + len(key + "[", start_index)
    end = sgf_string.find("]", start)
    data = sgf_string[start:end]
    return data, end + 1