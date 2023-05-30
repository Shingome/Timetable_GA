from deap import base, algorithms
from deap import creator
from deap import tools
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def split_to_days(timetable):
    return np.array(np.array_split(timetable, DAYS))


def split_to_hour(day):
    return day[0], *np.array_split(day[1:], INTERVALS / 2)


def get_pairs(timetable):
    timetable = split_to_days(timetable)[::, 1:]
    return np.array_split(timetable.flatten(), timetable.size // 2)


def get_better_pairs(timetable):
    timetable = get_better_hours(timetable)
    return np.array_split(timetable, timetable.size // 2)


def get_best_pairs(timetable):
    timetable = split_to_days(timetable)[::, 1:-4]
    return np.array_split(timetable, timetable.size // 2)


def get_better_hours(timetable):
    return split_to_days(timetable)[::, 1:-2].flatten()


def get_best_hours(timetable):
    return split_to_days(timetable)[::, 1:-4].flatten()


def get_last_hours(timetable):
    return split_to_days(timetable)[::, -2:].flatten()


def get_first_hours(timetable):
    return split_to_days(timetable)[::, :1].flatten()


def table_to_str(timetable):
    return np.array_split(tuple(SUBJECT_NAMES[i - 1] if i != 0 else "" for i in timetable), DAYS)


def show(individual):
    for i in table_to_str(individual):
        print(i)

def evaluateFunction(individual):
    return all_hours(individual) + \
           first_hours(individual) + \
           full_subject(individual) + \
           empty_hours(individual) + \
           last_hours(individual),


def all_hours(timetable):
    fine = [0] + [0 for _ in range(len(NEEDS))]
    for i in timetable:
        fine[i] += 1
    return np.sum(np.abs(np.asarray(fine)[1:] - np.asarray(NEEDS))) * 3


def first_hours(timetable):
    return len(list(filter(lambda x: int(x != 0), get_first_hours(timetable)))) * 2


def last_hours(timetable):
    return len(list(filter(lambda x: int(x != 0), get_last_hours(timetable)))) * 2


def empty_hours(timetable):
    return len(list(filter(lambda x: int(x == 0), get_best_hours(timetable))))


def full_subject(timetable):
    return len(list(filter(lambda x: x[0] != x[1], get_pairs(timetable))))


if __name__ == "__main__":
    DAYS = 5
    INTERVALS = 5 * 2 + 1
    SUBJECT_NAMES = ["Математика", "ИСТОРИЯ", "БИОЛОГИЯ", "РУССКИЙ"]
    SUBJECTS = [1, 2, 3, 4]
    NEEDS = [9, 9, 9, 9]

    POPULATION_SIZE = 2000
    P_CROSSOVER = 0.9
    P_MUTATION = 0.1
    MAX_GENERATIONS = 75
    TOURNAMENTSIZE = 3
    HALL_OF_FAME_SIZE = 1

    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("randomLectures", random.randint, 0, len(SUBJECTS) - 1)
    toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.randomLectures,
                     INTERVALS * DAYS)
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

    population = toolbox.populationCreator(n=POPULATION_SIZE)
    toolbox.register("evaluate", evaluateFunction)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENTSIZE)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(SUBJECTS), indpb=1.0 / len(SUBJECTS))

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    population, logbook = algorithms.eaSimple(population, toolbox,
                                              cxpb=P_CROSSOVER,
                                              mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS,
                                              stats=stats,
                                              halloffame=hof,
                                              verbose=True)

    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Поколение')
    plt.ylabel('Макс/средняя приспособленность')
    plt.title('Зависимость максимальной и средней приспособленности от поколения')
    plt.show()

    population = list(sorted(population, key=evaluateFunction))

    timetable = hof[0]

    print("Fitness:", evaluateFunction(timetable))
    print("All hours:", all_hours(timetable))
    print("First hours:", first_hours(timetable))
    print("Empty hours:", empty_hours(timetable))
    print("Full subject:", full_subject(timetable))
    print("Last hours:", last_hours(timetable))
    # print("best time:", best_time(timetable))

    timetable = pd.DataFrame(table_to_str(timetable), index="пн вт ср чт пт".split(" "))

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=timetable.values, colLabels=timetable.columns, loc='center')
    fig.tight_layout()
    plt.show()
