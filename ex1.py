import numpy as np
import random
import math
import self as self

#class Circle1
# This class represents a circle by 2 points (point on the circle and the center point) and the radius of the circle
# plus means does the circle have pluses or minuses inside
class Circle1:
    def __init__(self, alf, x1, xCenter, y1, yCenter, plus, radius):
        self.alf = alf
        self.x1 = x1
        self.xCenter = xCenter
        self.y1 = y1
        self.yCenter = yCenter
        self.plus = plus
        self.radius = radius

#class Rect
# This class represents a rectangle by 2 points
# plus means does the rectangle have pluses or minuses inside
class Rect:
    def __init__(self, alf, x1, x2, y1, y2, plus):
        self.alf = alf
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.plus = plus
#Circle
# this function gets 65 points, builds all possible circles
# the function checks if the sign inside the circle is compatible with the gender (+ inside - gender = 1 , - inside - gender = 2)
def Circle(setOfPionts,Weight,Temp,Gender,Heart):
    error = 1
    # creating all the possible circles
    for i in range(len(setOfPionts)):
        x1, y1 = Temp[i], Heart[i]
        for j in range(i+1,len(setOfPionts)):
            xCenter, yCenter = Temp[j], Heart[j]
            # calculate circle's radius
            radius = math.sqrt((x1 - xCenter) ** 2 + (y1 - yCenter) ** 2)
            tempError = 0
            #for +(1) inside the rectangle
            for t in range(len(Gender)):
                # check if the point inside the circle and the gender = 2 - if so it is error
                if (Temp[t] - xCenter) ** 2 + (Heart[t] - yCenter) ** 2 <= radius*radius:
                    if Gender[t] == 2:
                        tempError += Weight[t]

                # the point outside the circle and the gender = 1 - it's error
                else:
                    if Gender[t] == 1:
                        tempError += Weight[t]
            if tempError < error:
                error = tempError
                xr1, xr2, yr1, yr2 = x1, xCenter, y1, yCenter
                plus = 1
                rad = radius

            tempError = 0
            # for -(2) inside the rectangle
            for t in range(len(Gender)):
                # check if the point inside the circle and the gender = 1 - if so it is error
                if (Temp[t] - xCenter) ** 2 + (Heart[t] - yCenter) ** 2 <= radius*radius:
                    if Gender[t] == 1:
                        tempError += Weight[t]

                # the point outside the circle and the gender = 2 - it's error
                else:
                    if Gender[t] == 2:
                        tempError += Weight[t]
            if tempError < error:
                error = tempError
                xr1, xr2, yr1, yr2 = x1, xCenter, y1, yCenter
                plus = -1
                rad = radius

    # return the best circle (with min error), it's error and the sign inside it
    return error,xr1, xr2, yr1, yr2, plus, rad

#check_points_of_rectangle
# this function receives 2 points and checks which is the minimum and the maximum
def check_points_of_rectangle(x1, x2, y1, y2):
    xMin, xMax, yMin, yMax = 0, 0, 0, 0
    if x1 > x2 and y1 > y2:
        xMax, xMin, yMax, yMin = x1, x2, y1, y2
    elif x1 > x2 and y2 > y1:
        xMax, xMin, yMax, yMin = x1, x2, y2, y1
    elif x2 > x1 and y1 > y2:
        xMax, xMin, yMax, yMin = x2, x1, y1, y2
    else:
        xMax, xMin, yMax, yMin = x2, x1, y2, y1

    return xMin, xMax, yMin, yMax

#Rectangle
# this function gets 65 points, builds all possible rectangles
# the function checks if the sign inside the rectangle is compatible with the gender (+ inside - gender = 1 , - inside - gender = 2)
def Rectangle(setOfPionts,Weight,Temp,Gender,Heart):
    error = 1
    # creating all the possible rectangles
    for i in range(len(setOfPionts)):
        x1, y1 = Temp[i], Heart[i]
        for j in range(i+1,len(setOfPionts)):
            x2, y2 = Temp[j], Heart[j]
            xMin, xMax, yMin, yMax = check_points_of_rectangle(x1, x2, y1, y2)

            tempError = 0
            # for +(1) inside the rectangle
            for t in range(len(Gender)):
                # check if the point inside the rectangle and the gender = 2 - if so it is error
                if xMin <= Temp[t] <= xMax and yMin <= Heart[t] <= yMax:
                    if Gender[t] == 2:
                        tempError += Weight[t]

                # the point outside the rectangle and the gender = 1 - it's error
                else:
                    if Gender[t] == 1:
                        tempError += Weight[t]
            if tempError < error:
                error = tempError
                xr1, xr2, yr1, yr2 = xMin, xMax, yMin, yMax
                plus = 1

            tempError = 0
            # for -(2) inside the rectangle
            for t in range(len(Gender)):
                # check if the point inside the rectangle and the gender = 1 - if so it is error
                if xMin <= Temp[t] <= xMax and yMin <= Heart[t] <= yMax:
                    if Gender[t] == 1:
                        tempError += Weight[t]

                # the point outside the rectangle and the gender = 2 - it's error
                else:
                    if Gender[t] == 2:
                        tempError += Weight[t]
            if tempError < error:
                error = tempError
                xr1, xr2, yr1, yr2 = xMin, xMax, yMin, yMax
                plus = -1

    # return the best rectangle (with min error), it's error and the sign inside it
    return error,xr1, xr2, yr1, yr2, plus

# AdaBoost
# this function receives rectangles / circles for learning - the training set
def AdaBoost(R,func,r,x):
    Weight, rect, cir = [], [], []
    # initialize the weight set
    for i in range(len(R)):
        Weight.append(1 / 65)

    # x = 1 - it is rectangle
    if x==1:
        for i in range(r):
            e1, xMin, xMax, yMin, yMax, qqq = func(R, Weight, TempTraining, GenderTraining, HeartTraining)
            alfa = 0.5 * np.log((1-e1)/e1)
            rect.append(Rect(alfa, xMin, xMax, yMin, yMax, qqq))
            for j in range(len(R)):

                # check if the point inside the rectangle  and gender doesn't match the sign inside - if so update weight
                if xMin <= TempTraining[j] <= xMax and yMin <= HeartTraining[j] <= yMax:
                    if (GenderTraining[j] == 2 and qqq == 1) or (GenderTraining[j] == 1 and qqq == -1):
                        Weight[j] = Weight[j]*math.exp(alfa)
                    else:
                        Weight[j] = Weight[j]*math.exp(-alfa)

                # the point outside the rectangle  and gender doesn't match the sign inside - if so update weight
                else:
                    if (GenderTraining[j] == 2 and qqq == -1) or (GenderTraining[j] == 1 and qqq == 1):
                        Weight[j] = Weight[j] * math.exp(alfa)
                    else:
                        Weight[j] = Weight[j] * math.exp(-alfa)
            sum = 0
            for j in range(len(R)):
                sum += Weight[j]
            # normalize weights
            for j in range(len(R)):
                Weight[j] = Weight[j] / sum
        return rect

    #  it is circle
    else:
        for i in range(r):
            e1, x1, xCenter, y1, yCenter, qqq,rad = func(R, Weight, TempTraining, GenderTraining, HeartTraining)
            alfa = 0.5 * np.log((1-e1)/e1)
            cir.append(Circle1(alfa, x1, xCenter, y1, yCenter, qqq,rad))
            for j in range(len(R)):
                # check if the point inside the circle  and gender doesn't match the sign inside - if so update weight
                if (TempTraining[j] - xCenter) ** 2 + (HeartTraining[j] - yCenter) ** 2 <= rad**2:
                    if (GenderTraining[j] == 2 and qqq == 1) or (GenderTraining[j] == 1 and qqq == -1):
                        Weight[j] = Weight[j]*math.exp(alfa)
                    else:
                        Weight[j] = Weight[j]*math.exp(-alfa)

                # the point outside the circle  and gender doesn't match the sign inside - if so update weight
                else:
                    if (GenderTraining[j] == 2 and qqq == -1) or (GenderTraining[j] == 1 and qqq == 1):
                        Weight[j] = Weight[j] * math.exp(alfa)
                    else:
                        Weight[j] = Weight[j] * math.exp(-alfa)
            sum = 0
            for j in range(len(R)):
                sum += Weight[j]
            # normalize weights
            for j in range(len(R)):
                Weight[j] = Weight[j] / sum
        return cir

# main for running AdaBoost
with open('set of points.txt', 'r') as f:
    # reading the text file and insert it into array
    lines = f.readlines()

# this part is for the hypothesis of axis-parallel rectangles
print("results for rectangles:")
for r in range(1,9):
    sumAllH = 0
    sumAllR = 0
    for j in range(1,101):
        #randomly mixing text lines
        random.shuffle(lines)
        #dividing the mixed array into 2 new arrays (training and testing)
        training = lines[:65]
        testing = lines[65:]

        TempTesting, GenderTesting, HeartTesting, TempTraining, GenderTraining, HeartTraining = [], [], [], [], [], []
        #inserting data in to testing and training arrays
        for i in range(len(testing)):
            temp1 = testing[i].split(" ")
            temp2 = training[i].split(" ")
            first1 = float(temp1[0])
            first2 = float(temp2[0])
            TempTesting.append(first1)
            TempTraining.append(first2)
            if first1 >= 100:
                second1 = float(temp1[3])
                third1 = float(temp1[7])
            else:
                second1 = float(temp1[4])
                third1 = float(temp1[8])
            GenderTesting.append(second1)
            HeartTesting.append(third1)
            if first2 >= 100:
                second2 = float(temp2[3])
                third2 = float(temp2[7])
            else:
                second2 = float(temp2[4])
                third2 = float(temp2[8])
            GenderTraining.append(second2)
            HeartTraining.append(third2)

        #sending the training array and the rectangle to AdaBoost
        result = AdaBoost(training, Rectangle, r, 1)

        #sum the errors on test
        for i in range(len(testing)):
            sumHx = 0
            for n in range(len(result)):
                if result[n].plus == 1:
                    if result[n].x1 <= TempTesting[i] <= result[n].x2 and result[n].y1 <= HeartTesting[i] <= result[n].y2:
                        sumHx += result[n].alf
                    else:
                        sumHx -= result[n].alf
                else:
                    if result[n].x1 <= TempTesting[i] <= result[n].x2 and result[n].y1 <= HeartTesting[i] <= result[n].y2:
                        sumHx -= result[n].alf
                    else:
                        sumHx += result[n].alf
            if (sumHx > 0 and GenderTesting[i] == 2) or (sumHx < 0 and GenderTesting[i] == 1):
                sumAllH += 1


        #sum the errors on training
        for i in range(len(training)):
            sumHx = 0
            for n in range(len(result)):
                if result[n].plus == 1:
                    if result[n].x1 <= TempTraining[i] <= result[n].x2 and result[n].y1 <= HeartTraining[i] <= result[n].y2:
                        sumHx += result[n].alf
                    else:
                        sumHx -= result[n].alf
                else:
                    if result[n].x1 <= TempTraining[i] <= result[n].x2 and result[n].y1 <= HeartTraining[i] <= result[n].y2:
                        sumHx -= result[n].alf
                    else:
                        sumHx += result[n].alf
            if (sumHx > 0 and GenderTraining[i] == 2) or (sumHx < 0 and GenderTraining[i] == 1):
                sumAllR += 1
    print("for the set testing with " , r , "rounds in adaBoost: ", (sumAllH/100)/65)
    print("for the set training with " , r , "rounds in adaBoost: ", (sumAllR/100)/65)
    print("")

print("results for circles:")
# this part is for the hypothesis of circles
for r in range(1,9):
    sumAllH = 0
    sumAllR = 0
    for j in range(1,101):
        # randomly mixing text lines
        random.shuffle(lines)
        # dividing the mixed array into 2 new arrays (training and testing)
        training = lines[:65]
        testing = lines[65:]

        TempTesting, GenderTesting, HeartTesting, TempTraining, GenderTraining, HeartTraining = [], [], [], [], [], []
        # inserting data in to testing and training arrays
        for i in range(len(testing)):
            temp1 = testing[i].split(" ")
            temp2 = training[i].split(" ")
            first1 = float(temp1[0])
            first2 = float(temp2[0])
            TempTesting.append(first1)
            TempTraining.append(first2)
            if first1 >= 100:
                second1 = float(temp1[3])
                third1 = float(temp1[7])
            else:
                second1 = float(temp1[4])
                third1 = float(temp1[8])
            GenderTesting.append(second1)
            HeartTesting.append(third1)
            if first2 >= 100:
                second2 = float(temp2[3])
                third2 = float(temp2[7])
            else:
                second2 = float(temp2[4])
                third2 = float(temp2[8])
            GenderTraining.append(second2)
            HeartTraining.append(third2)

        # sending the training array and the circle to AdaBoost
        result = AdaBoost(training, Circle, r, 2)

        # sum the errors on test
        for i in range(len(testing)):
            sumHx = 0
            for n in range(len(result)):
                if result[n].plus == 1:
                    if (TempTesting[i] - result[n].xCenter) ** 2 + (HeartTesting[i] - result[n].yCenter) ** 2 <= result[n].radius**2:
                        sumHx += result[n].alf
                    else:
                        sumHx -= result[n].alf
                else:
                    if (TempTesting[i] - result[n].xCenter) ** 2 + (HeartTesting[i] - result[n].yCenter) ** 2 <= result[n].radius**2:
                        sumHx -= result[n].alf
                    else:
                        sumHx += result[n].alf
            if (sumHx > 0 and GenderTesting[i] == 2) or (sumHx < 0 and GenderTesting[i] == 1):
                sumAllH += 1

        # sum the errors on training
        for i in range(len(training)):
            sumHx = 0
            for n in range(len(result)):
                if result[n].plus == 1:
                    if (TempTraining[i] - result[n].xCenter) ** 2 + (HeartTraining[i] - result[n].yCenter) ** 2 <= result[n].radius**2:
                        sumHx += result[n].alf
                    else:
                        sumHx -= result[n].alf
                else:
                    if (TempTraining[i] - result[n].xCenter) ** 2 + (HeartTraining[i] - result[n].yCenter) ** 2 <= result[n].radius**2:
                        sumHx -= result[n].alf
                    else:
                        sumHx += result[n].alf
            if (sumHx > 0 and GenderTraining[i] == 2) or (sumHx < 0 and GenderTraining[i] == 1):
                sumAllR += 1
    print("for the set testing with ", r, "rounds in adaBoost: ", (sumAllH/100)/65)
    print("for the set training with ", r, "rounds in adaBoost: ", (sumAllR/100)/65)
    print("")