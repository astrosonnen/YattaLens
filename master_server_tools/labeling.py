from math import floor


def coords2name(rr, dd):
    flag=0
    rr1=floor(rr/15.)
    rr2=floor((rr/15.-rr1)*60)
    rr3=(((rr/15.-rr1)*60) - rr2)*60

    if(dd<0):
        flag=1
        dd=-1*dd

    dd1=floor(dd)
    dd2=floor((dd-dd1)*60)
    dd3=(((dd-dd1)*60)-dd2)*60

    if(flag):
        return "%02d%02d%02d"%(rr1,rr2,round(rr3)), "-%02d%02d%02d"%(dd1,dd2,round(dd3))
    else:
        return "%02d%02d%02d"%(rr1,rr2,round(rr3)), "+%02d%02d%02d"%(dd1,dd2,round(dd3))


