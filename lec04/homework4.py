
def next_birthday(date, birthdays):
    '''
    Find the next birthday after the given date.

    @param:
    date - a tuple of two integers specifying (month, day)
    birthdays - a dict mapping from date tuples to lists of names, for example,
      birthdays[(1,10)] = list of all people with birthdays on January 10.

    @return:
    birthday - the next day, after given date, on which somebody has a birthday
    list_of_names - list of all people with birthdays on that date
    '''
    month = date[0]
    day = date[1]

    sorted_keys = sorted(birthdays.keys())

    i = 0
    while i < len(sorted_keys) and sorted_keys[i][0] < month:
        i += 1

    while (i < len(sorted_keys) and
           sorted_keys[i][0] == month and
           sorted_keys[i][1] <= day):
        i += 1
    if i == len(sorted_keys):
        return sorted_keys[0], birthdays[sorted_keys[0]]
    return sorted_keys[i], birthdays[sorted_keys[i]]
