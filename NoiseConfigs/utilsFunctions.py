import math

def yellow_bg(text):
    return f"\033[43m{text}\033[0m"

class UtilsFunc:
    FREQUENCY_GH = 5

    @staticmethod
    def distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    @staticmethod
    def path_loss_km_ghz(d_km, f_ghz, n=2):
        """
        Calculate Path Loss in decibels (dB) using distance in km, frequency in GHz, and path loss exponent

        :param d_km: Distance between transmitter and receiver (kilometers)
        :param f_ghz: Frequency of transmitted wave (GHz)
        :param n: Path loss exponent (default: 2 for free space)
        :return: Path Loss value in dB
        """
        if d_km < 0 or f_ghz <= 0:
            raise ValueError(f"d_km and f_ghz must be positive values : {d_km}, {f_ghz}")
        elif d_km == 0:
            return 0
        c = 3e8
        pi = math.pi

        PL = 10 * n * math.log10(d_km) + 10 * n * math.log10(f_ghz) + 10 * n * math.log10(4 * pi / c) + 10 * n * (3+9)
        return PL


# Example usage
if __name__ == "__main__":
    x = UtilsFunc.distance(4214.90, 1932.26, 6000, 1500)
    print(x)
    print(UtilsFunc().path_loss_km_ghz(x/1000, 5, 0.2))
