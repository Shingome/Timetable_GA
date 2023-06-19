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
    groups = split_by_groups(individual)
    return sum(evaluateGroup(groups[i], i) for i in range(GROUPS)) + \
           teacher_crossing(individual) + \
           cabinet_crossing(individual),


def split_by_groups(individual):
    return np.reshape(individual, (GROUPS, -1))


# Evaluate group
def evaluateGroup(group, group_id):
    subjects, teachers, cabinets = group_preprocessing(group)
    return evaluateSubject(subjects, group_id) \
        + teacher_subject(subjects, teachers, group_id) \
        + empty_cabinet(cabinets, subjects)


def union(arr):
    return sorted(set(itertools.chain(*arr)))


def zero_timetable(arr):
    return np.zeros((len(arr), DAYS * HOURS))


def teacher_crossing(individual):
    all_teachers = union(TEACHERS)
    timetable = zero_timetable(all_teachers)

    for group_id, group in zip(range(GROUPS), split_by_groups(individual)):
        _, teachers, _ = group_preprocessing(group)
        for i, teacher in zip(range(GROUP_SIZE), teachers):
            timetable[all_teachers.index(TEACHERS[group_id][teacher])][i] += 1

    return np.sum(timetable[timetable > 1]) * 500


def cabinet_crossing(individual):
    all_cabinets = union(CABINETS)
    timetable = zero_timetable(all_cabinets)

    for group_id, group in zip(range(GROUPS), split_by_groups(individual)):
        _, _, cabinets = group_preprocessing(group)
        for i, cabinet in zip(range(GROUP_SIZE), cabinets):
            timetable[all_cabinets.index(CABINETS[group_id][cabinet])][i] += 1

    return np.sum(timetable[timetable > 1]) * 500


def group_preprocessing(individual):
    individual = np.array(individual)
    subjects = individual[::3]
    teachers = individual[1::3]
    cabinets = individual[2::3]

    return subjects, teachers, cabinets


def teacher_subject(subjects, teachers, group_id):
    return sum(
        (int(subjects[i] not in TEACHER_SUBJECT_NUM[group_id][teachers[i]]) for i in range(len(subjects)))) * 1000


def empty_cabinet(cabinets, subjects):
    return len(list(filter(lambda x: (x[0] != 0) if x[1] == 0 else False, zip(cabinets, subjects))))


# Evaluate Timetable of subjects
def evaluateSubject(timetable_subjects, group_id):
    return all_hours(timetable_subjects, group_id) + \
        first_hours(timetable_subjects) + \
        full_subject(timetable_subjects) + \
        empty_hours(timetable_subjects) + \
        last_hours(timetable_subjects)


def all_hours(subjects, group_id):
    needs = NEEDS[group_id]
    fine = [0 for _ in range(len(needs))]
    for i in subjects:
        fine[i] += 1
    # return np.sum(np.abs(np.asarray(fine)[1:] - np.asarray(needs[1:]))) * 1000
    return len(list(filter(lambda x: x[0] != x[1], zip(fine[1:], needs[1:])))) * 1000


def first_hours(subjects):
    return len(list(filter(lambda x: int(x != 0), get_first_hours(subjects))))


def last_hours(subjects):
    return len(list(filter(lambda x: int(x != 0), get_last_hours(subjects))))


def empty_hours(subjects):
    return len(list(filter(lambda x: int(x == 0), get_best_hours(subjects))))


def full_subject(subjects):
    return len(list(filter(lambda x: x[0] != x[1], get_pairs(subjects)))) * 10


# Create individual
def wrapper(func, *args):
    def wrapped_func():
        return func(*args)

    return wrapped_func


def group_random_fun():
    return np.transpose(list(list(wrapper(random.randint, *bounds)
                                  for bounds in ((0, len(sphere[i]) - 1)
                                                 for i in range(GROUPS)))
                             for sphere in (SUBJECTS, TEACHERS, CABINETS)))


def individualCreator(container):
    return \
        container(itertools.chain(*[tools.initCycle(list, func, GROUP_SIZE // 3) for func in group_random_fun()]))


# Mutation Func
def ownMutUniformInt(individual):
    def repeat(p_object, size):
        return itertools.chain(*(p_object for _ in range(size // len(p_object))))

    individual = split_by_groups(individual)

    for group in range(GROUPS):
        for i, type in zip(range(GROUP_SIZE), repeat((0, 1, 2), GROUP_SIZE)):
            sphere_size = len((SUBJECTS, TEACHERS, CABINETS)[type][group])
            if random.random() < (1 / sphere_size):
                individual[group][i] = random.randint(0, sphere_size - 1)

    return creator.Individual(individual.flatten()),


if __name__ == "__main__":
    # Day parameters
    DAYS = 5
    HOURS = 5 * 2 + 1

    # Teacher parameters
    TEACHERS = [
        ["", "БРЫЛЕВА А.А.", "СУТОВИЧ С.Г.", "ШАНДРИКОВ А.В.", "ШАППО М.М.", "КОЛЕСНИКОВИЧ М.В."],
        ["", "БРЫЛЕВА А.А.", "ГОРОШИН В.Б.", "ВЕЛИКОВ А.С."]
    ]
    TEACHER_SUBJECT = [
        [
            [""], ["МАТЕМАТИКА", "РУССКИЙ"], ["ИСТОРИЯ"], ["БИОЛОГИЯ"], ["РУССКИЙ"], ["ИСТОРИЯ"]
        ],
        [
            [""], ["МАТЕМАТИКА", "РУССКИЙ"], ["ЧЕРЧЕНИЕ"], ["ГЕОГРАФИЯ"]
        ]
    ]
    TEACHER_SUBJECT_NUM = [
        [
            [0], [1, 4], [2], [3], [4], [2]
        ],
        [
            [0], [2, 3], [4], [1]
        ]
    ]

    # Cabinet parameters
    CABINETS = [
        ["", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8"],
        ["", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7"],
    ]

    # Subject parameters
    SUBJECTS = [
        ["", "МАТЕМАТИКА", "ИСТОРИЯ", "БИОЛОГИЯ", "РУССКИЙ"],
        ["", "ГЕОГРАФИЯ", "МАТЕМАТИКА", "РУССКИЙ", "ЧЕРЧЕНИЕ"],
    ]
    NEEDS = [
        [-1, 9, 9, 9, 9],
        [-1, 9, 9, 9, 9],
    ]

    # Hyper parameters
    GROUPS = 2
    GROUP_SIZE = HOURS * DAYS * 3
    INDIVIDUAL_SIZE = GROUP_SIZE * GROUPS
    POPULATION_SIZE = 3000
    P_CROSSOVER = 0.9
    P_MUTATION = 0.1
    MAX_GENERATIONS = 80
    TOURNAMENT_SIZE = 5
    HALL_OF_FAME_SIZE = 1

    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individualCreator", individualCreator, creator.Individual)

    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

    population = toolbox.populationCreator(n=POPULATION_SIZE)
    toolbox.register("evaluate", evaluateFunction)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", ownMutUniformInt)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    population, logbook = algorithms.eaSimple(population, toolbox,
                                              cxpb=P_CROSSOVER,
                                              mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS,
                                              stats=stats,
                                              halloffame=hof,
                                              verbose=True)

    maxFitnessValues, meanFitnessValues, minFitnessValues = logbook.select("max", "avg", "min")

    plt.plot(maxFitnessValues, color='red')
    plt.plot(minFitnessValues, color='yellow')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Поколение')
    plt.ylabel('Макс/Мин/Ср приспособленность')
    plt.title('Зависимость максимальной, минимальной и средней приспособленности от поколения')
    plt.show()

    population = list(sorted(population, key=evaluateFunction))


    def ind_to_str(individual):
        def sphere_to_str(sphere, sphere_str):
            return list(itertools.chain(*([sphere_str[group_id][i] for i in group]
                                          for group, group_id in zip(np.array_split(sphere, GROUPS), range(GROUPS)))))

        individual = zip(sphere_to_str(sphere, sphere_str)
                         for sphere, sphere_str in zip(group_preprocessing(individual), (SUBJECTS, TEACHERS, CABINETS)))

        return list(itertools.chain(*individual))


    def ind_to_table(individual):
        individual = split_by_groups(list(zip(*ind_to_str(individual))))
        return list(np.array_split(group, DAYS) for group in individual)


    ind = hof[0]
    table = ind_to_table(hof[0])
    for i in range(len(table)):
        pd.DataFrame(table[i]).to_excel(f"group_{i}.xlsx")

    groups = split_by_groups(ind)
    print("\nFitness:", evaluateFunction(ind))
    print("Teacher_crossing:", teacher_crossing(ind))
    print("Cabinet_crossing:", cabinet_crossing(ind))
    for i in range(len(groups)):
        subjects, teachers, cabinets = group_preprocessing(groups[i])

        print(f"\ngroup_{i}")
        print("All hours:", all_hours(subjects, i))
        print("First hours:", first_hours(subjects))
        print("Empty hours:", empty_hours(subjects))
        print("Full subject:", full_subject(subjects))
        print("Last hours:", last_hours(subjects))
        print("Teacher-Subject: ", teacher_subject(subjects, teachers, i))
        print("Empty cabinets: ", empty_cabinet(cabinets, subjects))
