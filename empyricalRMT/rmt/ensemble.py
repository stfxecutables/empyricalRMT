import numpy as np


class Poisson:
    @staticmethod
    def spacing_distribution(unfolded: np.array) -> np.array:
        spacings = unfolded[1:] - unfolded[:-1]
        s = np.linspace(spacings.min(), spacings.max(), 10000)
        return np.exp(-s)

    @staticmethod
    def spectral_rigidity(
        unfolded: np.array, L_min: float = 0.5, L_max: float = 20, num_L: int = 50
    ) -> np.array:
        spacings = unfolded[1:] - unfolded[:-1]
        L = np.linspace(L_min, L_max, num_L)
        return L / 15 / 2

    @staticmethod
    def level_variance(
        unfolded: np.array, L_min: float = 0.5, L_max: float = 20, num_L: int = 50
    ) -> np.array:
        spacings = unfolded[1:] - unfolded[:-1]
        L = np.linspace(L_min, L_max, num_L)
        s = L / np.mean(spacings)
        return s / 2


class GOE:
    @staticmethod
    def spacing_distribution(unfolded: np.array, n_points: int = 1000) -> np.array:
        """return expected spacings over the range [spacings.min(), spacings.max()], where
        `spacings` are the spacings calculated from `unfolded`
        """
        spacings = unfolded[1:] - unfolded[:-1]
        s = np.linspace(spacings.min(), spacings.max(), n_points)
        p = np.pi
        return ((p * s) / 2) * np.exp(-(p / 4) * s * s)

    @staticmethod
    def spectral_rigidity(
        unfolded: np.array, L_min: float = 0.5, L_max: float = 20, num_L: int = 50
    ) -> np.array:
        spacings = unfolded[1:] - unfolded[:-1]
        L = np.linspace(L_min, L_max, num_L)
        s = L / np.mean(spacings)
        p, y = np.pi, np.euler_gamma
        return (1 / (p ** 2)) * (np.log(2 * p * s) + y - 5 / 4 - (p ** 2) / 8)

    @staticmethod
    def level_variance(
        unfolded: np.array, L_min: float = 0.5, L_max: float = 20, num_L: int = 50
    ) -> np.array:
        spacings = unfolded[1:] - unfolded[:-1]
        L = np.linspace(L_min, L_max, num_L)
        s = L / np.mean(spacings)
        p = np.pi
        return (2 / (p ** 2)) * (np.log(2 * p * s) + np.euler_gamma + 1 - (p ** 2) / 8)


class GUE:
    @staticmethod
    def spacing_distribution(unfolded: np.array) -> np.array:
        spacings = unfolded[1:] - unfolded[:-1]
        s = np.linspace(spacings.min(), spacings.max(), 10000)
        p = np.pi
        return (32 / p ** 2) * (s * s) * np.exp(-(4 * s * s) / p)

    @staticmethod
    def spectral_rigidity(
        unfolded: np.array, L_min: float = 0.5, L_max: float = 20, num_L: int = 50
    ) -> np.array:
        spacings = unfolded[1:] - unfolded[:-1]
        L = np.linspace(L_min, L_max, num_L)
        s = L / np.mean(spacings)
        p = np.pi
        return (1 / (2 * (p ** 2))) * (np.log(2 * p * s) + np.euler_gamma - 5 / 4)

    @staticmethod
    def level_variance(
        unfolded: np.array, L_min: float = 0.5, L_max: float = 20, num_L: int = 50
    ) -> np.array:
        spacings = unfolded[1:] - unfolded[:-1]
        L = np.linspace(L_min, L_max, num_L)
        s = L / np.mean(spacings)
        p = np.pi
        return (1 / (p ** 2)) * (np.log(2 * p * s) + np.euler_gamma + 1)


class GSE:
    @staticmethod
    def spacing_distribution(unfolded: np.array) -> np.array:
        spacings = unfolded[1:] - unfolded[:-1]
        s = np.linspace(spacings.min(), spacings.max(), 10000)
        p = np.pi
        # (2 ** 18 / (3 ** 6 * p ** 3)) * (s ** 4) * np.exp(-((64 / (9 * p)) * (s * s)))
        return (
            (262144 / (729 * p ** 3)) * (s ** 4) * np.exp(-((64 / (9 * p)) * (s * s)))
        )

    @staticmethod
    def spectral_rigidity(
        unfolded: np.array, L_min: float = 0.5, L_max: float = 20, num_L: int = 50
    ) -> np.array:
        spacings = unfolded[1:] - unfolded[:-1]
        L = np.linspace(L_min, L_max, num_L)
        s = L / np.mean(spacings)
        p, y = np.pi, np.euler_gamma
        return (1 / (4 * (p ** 2))) * (np.log(4 * p * s) + y - 5 / 4 + (p ** 2) / 8)

    @staticmethod
    def level_variance(
        unfolded: np.array, L_min: float = 0.5, L_max: float = 20, num_L: int = 50
    ) -> np.array:
        spacings = unfolded[1:] - unfolded[:-1]
        L = np.linspace(L_min, L_max, num_L)
        s = L / np.mean(spacings)
        p, y = np.pi, np.euler_gamma
        return (1 / (2 * (p ** 2))) * (np.log(4 * p * s) + y + 1 + (p ** 2) / 8)
