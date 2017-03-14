import cv2

class Controls:
    def __init__(self, ranges):
        cv2.namedWindow('controls', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('controls', 900,0)
        print("Control length:",len(ranges))
        self.t = []
        self.controlCount = 0
        if isinstance(ranges[0], tuple) != True:
            raise TypeError('Argument not a list of tuples')
        for i in range(len(ranges)):
            self.t.append(ranges[i][1])
            cv2.createTrackbar("t%d" % (i), 'controls', ranges[i][0], ranges[i][2],self.TrackBarCallback)
            cv2.setTrackbarPos("t%d" % (i), 'controls',ranges[i][1])
            self.controlCount += 1
    def Set(self, number, value):
        if number >= self.controlCount:
            raise OverflowError("No such control")
        cv2.setTrackbarPos("t%d" % (number), 'controls',value)
        self.t[number] = value
    def TrackBarCallback(self, arg1):
        for i in range(self.controlCount):
            self.t[i] = cv2.getTrackbarPos("t%d" % (i), 'controls')
            #print("t%d has changed to %d" % (i, self.t[i]))


