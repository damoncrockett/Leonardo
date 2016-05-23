import pandas as pd
from PIL import Image
import os

infile = "/Users/damoncrockett/Desktop/Leo/local_path.csv"

df = pd.read_csv(infile)

thumb_side = 128

for i in range(len(df)):
	
	file = df.local_path.loc[i]
	
	try:
		im = Image.open(file)
		w = im.width
		h = im.height
		
		if w > 127 and h > 127:

			if w > h:
				im = im.crop((0,0,h,h))
				im.thumbnail((thumb_side,thumb_side),Image.ANTIALIAS)
				im.save(file)
			elif h > w:
				im = im.crop((0,h-w,w,h))
				im.thumbnail((thumb_side,thumb_side),Image.ANTIALIAS)
				im.save(file)
			elif h == w:
				im.thumbnail((thumb_side,thumb_side),Image.ANTIALIAS)
				im.save(file)
				
		else:
		
		    os.remove(file)
		    df = df.drop(i)

	except Exception as e:
		print file, e
		os.remove(file)
		df = df.drop(i)
		
df.reset_index(drop=True,inplace=True)
df.to_csv(infile,index=False)
            
