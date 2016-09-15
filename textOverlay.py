from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

img = Image.open("uploads/IMG_1399.jpg")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("Silom.ttf", 30)

x, y = 100, 100

for i in range(0, 10):
	x = i * 100
	y = i * 100
	draw.rectangle((x, y, x+20, y+35), fill='black')
	draw.text((x, y), str(i),(255,255,255), font=font)

img.save('sample-out.jpg')