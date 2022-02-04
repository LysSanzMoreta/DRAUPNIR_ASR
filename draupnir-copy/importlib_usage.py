from importlib import resources
...
# # read text file
# with resources.open_text('draupnir', 'somefile.txt') as fp:
#     txt = fp.read()
...
# or binary
with resources.open_binary('draupnir', 'LatentBlactamase.png') as fp:
    img = fp.read()

# or binary
with resources.open_binary('draupnir', 'MI.png') as fp2:
    img2 = fp2.read()