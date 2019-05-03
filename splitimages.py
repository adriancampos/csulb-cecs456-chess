import numpy as np
from PIL import Image
import glob, os
from io import BytesIO


"""# Config"""

debug_print_images = False
max_boards_to_process = 1000 # -1  # -1 for no limit.
remove_duplicates = True

inDirectory = "in"
outDirectory = "in_split"

"""## Chess/Image Utils"""

numRows, numCols = (8,8)

def is_dark_square(row, col):
  return (row+col)%2!=0

def get_piece_from_board(boardImg, row, col):
  boardSize = 400
  pieceSize = boardSize/8
  
  return boardImg.crop([
      col*pieceSize, # left
      row*pieceSize, # upper
      (col+1)*pieceSize, # right
      (row+1)*pieceSize # lower
  ])
  


def fen_to_arr(fenString, separator="/"):
  # creates a 2d array from a FEN string
  boardLabels = []
  for rowString in fenString.split(separator):
    # rowString looks something like "N3R1r1"
    row = []

    for c in rowString:
      # iterate through each char in rowString
      if c.isdigit():
        # if it's numeric, add that many blank spaces
        for _ in range(int(c)):
          row.append("0")
      else:
        # append the character itself
        row.append(c)


    boardLabels.append(row)

  return boardLabels
  
  
def slice_and_save_board(boardImg, boardLabels, outdir):
  # slices up boardImg and saves each piece as outdir/TYPE_COORDS.jpg
  os.makedirs(outdir, exist_ok=True)
  
  # store a set of already saved squares so we don't save the same thing multiple times
  saved_pieces = []
  
  global total_piece_count
  global duplicate_piece_count
  global nondup_piece_count

  for row in range(len(boardLabels)):
    for col in range(len(boardLabels[row])):
      piece_label = boardLabels[row][col]
      
      piece_and_board_bg = piece_label + str(is_dark_square(row, col))
      
      if piece_and_board_bg in saved_pieces:
        continue;
      saved_pieces.append(piece_and_board_bg)
      
      
      piece_image = get_piece_from_board(boardImg, row, col)
      

      if debug_print_images:
        # print images to console
        print(piece_label)
        
      total_piece_count += 1

      if (remove_duplicates and not is_image_a_duplicate(piece_image.tobytes())):
        nondup_piece_count += 1

        fileName = piece_label + "_" + ['a','b','c','d','e','f','g','h'][col] + str(8-row)

        piece_image.save(outdir + "/"+ fileName + ".jpg","JPEG")
      else:
        duplicate_piece_count += 1

"""## De-depulicate Utils"""

class NumpyObj:
    def __init__(self, np_array):
        self.np_array = np_array
    
    def __hash__(self):
        return hash(self.np_array)
    
    def get_np(self):
        return self.np_array
    
    def __eq__(self, other):
        return self.np_array == other.get_np()


      
duplicate_container = set()

# keep track of the number of duplicates
total_piece_count = 0
duplicate_piece_count = 0
nondup_piece_count = 0



def is_image_a_duplicate(image):
    if NumpyObj(image) not in duplicate_container:
        duplicate_container.add(NumpyObj(image))
        return False
    else:
        return True

def clear_duplicate_container():
    duplicate_container = set()
    
    total_piece_count = 0
    duplicate_piece_count = 0
    nondup_piece_count = 0

"""## Process images
Goes through all images and converts them to 64 individual images
"""

clear_duplicate_container()

for infile in glob.glob(inDirectory + "/**/*.jp*g", recursive=True):
    # take care of max_boards_to_process
    if max_boards_to_process == 0:
        break;
    max_boards_to_process -= 1
  
  
    # load image and name
    file, ext = os.path.splitext(infile)
    boardImg = Image.open(infile)
    fenString = os.path.basename(file)
    if debug_print_images:
      print(fenString)
    
    
    # convert name to label array
    boardLabelArr = fen_to_arr(fenString, separator="-")
    
    slice_and_save_board(boardImg, boardLabelArr, outDirectory + "/" + fenString)

print("total_piece_count", total_piece_count)
print("duplicate_piece_count", duplicate_piece_count)
print("nondup_piece_count", nondup_piece_count)
print("sanity check:", total_piece_count == duplicate_piece_count + nondup_piece_count)

# !zip -q -r in_split.zip in_split/
