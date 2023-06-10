from collections.abc import Iterable
from deap import base, algorithms
from deap import creator
from deap import tools
import itertools
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Support functions
def split_to_days(timetable):
    return np.array(np.array_split(timetable, DAYS))


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


def table_to_str(timetable, type):
    return np.array_split(tuple((SUBJECTS, TEACHERS, CABINETS)[type][i] for i in timetable), DAYS)


# Main evaluate
def evaluateFunction(individual):
    subjects, teachers, cabinets = individual_preprocessing(individual)
    return evaluateSubject(subjects) + teacher_subject(subjects, teachers) + empty_cabinet(cabinets, subjects),


def individual_preprocessing(individual):
    individual = np.array(individual)
    subjects = individual[::3]
    teachers = individual[1::3]
    cabinets = individual[2::3]

    return subjects, teachers, cabinets


# Evaluate Timetable of teachers
def evaluateTeacher(timetable_teachers):
    pass


def teacher_subject(subjects, teachers):
    return sum((int(subjects[i] not in TEACHER_SUBJECT_NUM[teachers[i]]) for i in range(len(subjects)))) * 1000


# Evaluate Timetable of cabinets
def evaluateCabinet(timetable_cabinets):
    pass


def empty_cabinet(cabinets, subjects):
    return len(list(filter(lambda x: (x[0] != 0) if x[1] == 0 else False, zip(cabinets, subjects))))


# Evaluate Timetable of subjects
def evaluateSubject(timetable_subjects):
    return all_hours(timetable_subjects) + \
        first_hours(timetable_subjects) + \
        full_subject(timetable_subjects) + \
        empty_hours(timetable_subjects) + \
        last_hours(timetable_subjects)


def all_hours(subjects):
    fine = [0 for _ in range(len(NEEDS))]
    for i in subjects:
        fine[i] += 1
    return np.sum(np.abs(np.asarray(fine)[1:] - np.asarray(NEEDS[1:]))) * 1000


def first_hours(subjects):
    return len(list(filter(lambda x: int(x != 0), get_first_hours(subjects))))


def last_hours(subjects):
    return len(list(filter(lambda x: int(x != 0), get_last_hours(subjects))))


def empty_hours(subjects):
    return len(list(filter(lambda x: int(x == 0), get_best_hours(subjects))))


def full_subject(subjects):
    return len(list(filter(lambda x: x[0] != x[1], get_pairs(subjects)))) * 10


# Mutation Func
def CycleMutUniformInt(individual, low, up, indpb):
    def repeat(p_object, size):
        return list(itertools.chain(*(p_object for _ in range(size // len(p_object)))))

    size = len(individual)

    if not isinstance(low, Iterable):
        raise IndexError("parameter low must be iterable")
    else:
        low = repeat(low, size)

    if not isinstance(up, Iterable):
        raise IndexError("parameter up must be iterable")
    else:
        up = repeat(up, size)

    for i, xl, xu in zip(range(size), low, up):
        if random.random() < indpb:
            individual[i] = random.randint(xl, xu)

    return individual,

# Create invidiaudual

def individualCreator():
    pass


if __name__ == "__main__":
    # Day parameters
    DAYS = 5
    HOURS = 5 * 2 + 1

    # Teacher parameters
    TEACHERS = ["", "БРЫЛЕВА А.А.", "СУТОВИЧ С.Г.", "ШАНДРИКОВ А.В.", "ШАППО М.М.", "КОЛЕСНИКОВИЧ М.В."]
    TEACHER_SUBJECT = [[""], ["МАТЕМАТИКА", "РУССКИЙ"], ["ИСТОРИЯ"], ["БИОЛОГИЯ"], ["РУССКИЙ"], ["ИСТОРИЯ"]]
    TEACHER_SUBJECT_NUM = [[0], [1, 4], [2], [3], [4], [2]]

    # Cabinet parameters
    CABINETS = ["", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8"]

    # Subject parameters
    SUBJECTS = ["", "МАТЕМАТИКА", "ИСТОРИЯ", "БИОЛОГИЯ", "РУССКИЙ"]
    NEEDS = [-1, 9, 9, 9, 9]

    # Hyper parameters
    GROUPS = 1
    GROUP_SIZE = HOURS * DAYS
    INDIVIDUAL_SIZE = GROUP_SIZE * GROUPS
    POPULATION_SIZE = 2000
    P_CROSSOVER = 0.9
    P_MUTATION = 0.1
    MAX_GENERATIONS = 1
    TOURNAMENT_SIZE = 3
    HALL_OF_FAME_SIZE = 1

    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("randomSubjects", random.randint, 0, len(SUBJECTS) - 1)
    toolbox.register("randomTeachers", random.randint, 0, len(TEACHERS) - 1)
    toolbox.register("randomCabinets", random.randint, 0, len(CABINETS) - 1)
    toolbox.register("individualCreator",
                     tools.initCycle,
                     creator.Individual,
                     (toolbox.randomSubjects, toolbox.randomTeachers, toolbox.randomCabinets),
                     INDIVIDUAL_SIZE)
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

    population = toolbox.populationCreator(n=POPULATION_SIZE)
    toolbox.register("evaluate", evaluateFunction)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate",
                     CycleMutUniformInt,
                     low=[0, 0, 0],
                     up=[len(SUBJECTS) - 1, len(TEACHERS) - 1, len(CABINETS) - 1],
                     indpb=1.0 / len(SUBJECTS))

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


    def tableSTR(individual):
        subjects, teachers, cabinets = individual_preprocessing(individual)
        subjects = (SUBJECTS[i] for i in subjects)
        teachers = (TEACHERS[i] for i in teachers)
        cabinets = (CABINETS[i] for i in cabinets)

        return np.array_split(np.array(list(zip(subjects, teachers, cabinets))).flatten(), DAYS)


    pd.DataFrame(tableSTR(hof[0])).to_excel("example.xlsx")

    timetable = hof[0]

    subject, teacher, cabinets = individual_preprocessing(timetable)

    print("Fitness:", evaluateFunction(timetable))
    print("All hours:", all_hours(subject))
    print("First hours:", first_hours(subject))
    print("Empty hours:", empty_hours(subject))
    print("Full subject:", full_subject(subject))
    print("Last hours:", last_hours(subject))
    print("Teacher-Subject: ", teacher_subject(subject, teacher))
    print("Empty cabinets: ", empty_cabinet(cabinets, subject))

    def show_timetable(timetable, type):
        timetable = pd.DataFrame(table_to_str(timetable, type), index="пн вт ср чт пт".split(" "))

        plt.rcParams["figure.figsize"] = [20, 10]
        plt.rcParams["figure.autolayout"] = True
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        table = ax.table(cellText=timetable.values, colLabels=timetable.columns, loc='center')
        fig.tight_layout()
        plt.show()

    show_timetable(subject, 0)
    show_timetable(teacher, 1)
    show_timetable(cabinets, 2)
