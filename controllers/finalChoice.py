import math
import random

from config import Config

def red_bg(text):
    return f"\033[41m{text}\033[0m"


class FinalChoice:

    def makeFinalChoice(self, finalCandidates, method):
        if len(finalCandidates) > 0:

            if method == Config.FinalDeciderMethod.RANDOM_CHOICE:
                return self.randomMethod(finalCandidates)

            elif method == Config.FinalDeciderMethod.FIRST_CHOICE:
                return finalCandidates[0]

            elif method == Config.FinalDeciderMethod.MIN_DISTANCE:
                return self.minDistanceMethod(finalCandidates)

    def randomMethod(self, attenuationList):
        valid_entries = self.calcValidEntries(attenuationList)
        # print(f"valid_entries:{valid_entries}")
        return random.choice(valid_entries)

    def calcValidEntries(self, attenuationList):
        valid_entries = []

        for i in range(0, len(attenuationList)):
            valid_entries.append(attenuationList[i])

        return valid_entries

    def minDistanceMethod(self, finalCandidates):
        x = self.minDistanceItem(finalCandidates)
        data = x[0]
        distance = x[1]

        # print(f"data : {data}, distance : {distance}")
        return data

    def calcMinDistance(self, zone, offloadedDevice):
        zoneX = zone.x
        zoneY = zone.y
        offloadedDeviceX = offloadedDevice.x
        offloadedDeviceY = offloadedDevice.y
        # print(red_bg(f"zoneX: {zoneX}, zoneY: {zoneY}, offloadedDeviceX: {offloadedDeviceX}, offloadedDeviceY: {offloadedDeviceY}"))
        distance = math.sqrt((offloadedDeviceX - zoneX) ** 2 + (offloadedDeviceY - zoneY) ** 2)
        return distance

    def minDistanceItem(self, finalCandidates):
        distances = []
        for i in range(0, len(finalCandidates)):
            distances.append(
                (finalCandidates[i], self.calcMinDistance(finalCandidates[i][0].zone, finalCandidates[i][1])))
        min_distance = min(distances, key=lambda x: x[1])
        return min_distance