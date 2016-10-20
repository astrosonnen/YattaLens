import numpy as np
import Image, ImageDraw, ImageFont
from yattatools import paper1_catalogs


known = ['020929-064312', '021411-040502', '021737-051329', '022346-053418', '022511-045433', '023307-043838', '142449-005321', '220506+014703', '221418+011036']
group = ['022410-033605', '222609+004141']

smallfont = ImageFont.truetype("/usr/local/texlive/2015/texmf-dist/fonts/truetype/public/dejavu/DejaVuSans-Bold.ttf", 8)
font = ImageFont.truetype("/usr/local/texlive/2015/texmf-dist/fonts/truetype/public/dejavu/DejaVuSans-Bold.ttf", 7)

npix = 101
ncol = 5

lowz_list = paper1_catalogs.get_success_list(subsample='LOWZ')
cmass_list = paper1_catalogs.get_success_list(subsample='CMASS')

gradeA = []
names = []
for cand in lowz_list + cmass_list:
    cand.get_grade()
    if cand.grade >= 2.5 and not cand.name in known:
	gradeA.append(cand)
	names.append(cand.name)

zipped = zip(names, gradeA)
zipped.sort()
snames, sgradeA = zip(*zipped)

nlens = len(gradeA)
nrow = nlens/ncol + min(1, nlens%ncol)

fullcomp = Image.new('RGB', (ncol*npix, nrow*npix), 'black')

for i in range(nlens):
    rowno = i/ncol
    colno = i%ncol

    im = sgradeA[i].make_postage_stamp()
    draw = ImageDraw.Draw(im)

    draw.text((3, 50), 'HSCJ'+sgradeA[i].name, font=smallfont, fill='white')        
    draw.text((30, 5), sgradeA[i].subsample.upper(), font=font, fill='white')        
    fullcomp.paste(im, (colno*npix, rowno*npix))

lens1 = paper1_catalogs.boss_candidate(name='090709+005648')
lens1.get_grade()
lens2 = paper1_catalogs.boss_candidate(name='142720+001916')
lens2.get_grade()
lens3 = paper1_catalogs.boss_candidate(name='022140-021020')
lens3.get_grade()


print lens1.name, lens1.votes
print lens2.name, lens2.votes
print lens3.name, lens3.votes

fullcomp.save('gradeA_collage.png')

