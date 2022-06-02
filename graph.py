from io import BytesIO
import base64
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import logging as log


def graphToImg(x,y,z,a,b,d,name,x_aixs,y_axis):
    try:
        fig = Figure()
        plt.title("regression")
        fig=plt.figure()
        global i 
        i=1
        axis = fig.add_subplot(1, 1, i)
        i=i+1
        buf = BytesIO()
        axis.set_title(name)
        axis.set_xlabel(x_aixs)
        axis.set_ylabel(y_axis)
        # axis.plot(x,y,c=z)
        axis.scatter(x,y,c=z)
        axis.scatter(a,b,c=d)
        # Save it to a temporary buffer.
        fig.savefig(buf, format="jpg")
        data = base64.b64encode(buf.getbuffer()).decode("utf-8")
        return data
    except Exception as e:
        log.error("this is error inside graph ",e)
        return "error"
    # Embed the result in the html output.
    # img = base64.b64encode(buf.getbuffer()).decode("ascii")
