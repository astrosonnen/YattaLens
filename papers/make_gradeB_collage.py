import numpy as np
import Image, ImageDraw, ImageFont
from yattatools import paper1_catalogs


showname = False

known = ['020929-064312', '021411-040502', '021737-051329', '022346-053418', '022511-045433', '023307-043838', '142449-005321', '220506+014703', '221418+011036']
group = ['022410-033605', '222609+004141']

smallfont = ImageFont.truetype("/usr/local/texlive/2015/texmf-dist/fonts/truetype/public/dejavu/DejaVuSans-Bold.ttf", 8)
font = ImageFont.truetype("/usr/local/texlive/2015/texmf-dist/fonts/truetype/public/dejavu/DejaVuSans-Bold.ttf", 7)

npix = 101
ncol = 8

lowz_list = paper1_catalogs.get_success_list(subsample='LOWZ')
cmass_list = paper1_catalogs.get_success_list(subsample='CMASS')

gradeB = []
names = []
for name in lowz_list + cmass_list:
    cand = paper1_catalogs.boss_candidate(name)
    cand.get_grade()
    if cand.grade >= 1.5 and cand.grade < 2.5 and not cand.name in known:
	gradeB.append(cand)
	names.append(name)

zipped = zip(names, gradeB)
zipped.sort()
snames, sgradeB = zip(*zipped)

nlens = len(gradeB)
print nlens

nrow = nlens/ncol + min(1, nlens%ncol)

fullcomp = Image.new('RGB', (ncol*npix, nrow*npix), 'black')

for i in range(nlens):
    rowno = i/ncol
    colno = i%ncol

    im = sgradeB[i].make_postage_stamp()

    if showname:
        draw = ImageDraw.Draw(im)
        draw.text((3, 80), 'HSCJ'+sgradeB[i].name, font=smallfont, fill='white')        
        draw.text((10, 5), sgradeB[i].subsample.upper(), font=font, fill='white')        
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

fullcomp.save('gradeB_collage.png')

