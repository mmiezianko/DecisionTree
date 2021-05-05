
def set_splitter(data, col, value):
# dzieli dane na podzbiory ze względu na wprowadzoną wartość (argument value)
    if isinstance(value, int) or isinstance(value, float): # dla wartości int lub float 
        splitter = lambda row : row[col] >= value
    else: # dla stringów
        splitter = lambda row : row[col] == value
    list1 = [row for row in data if splitter(row)] # pozbiór z daną wartością np. bouldering
    list2 = [row for row in data if not splitter(row)] # podzbiór z resztą wartości np. czasówki, lina
    return list1, list2


def unique_labels_counter(data): # zlicza wystąpienia dla danej etykiety
    results = {}
    for row in data:
        if not isinstance(row, int) and not isinstance(row, float):
            label = row[-1]
        else:
            label = row
        if label not in results: # operator bool czyli "jeśli lebel nie jest w bazie" to zainicjuj go (dodaj do bazy)
            results[label] = 0
        results[label] += 1 # dodaje do etykiety kolejne wystąpienie
    # zwraca słownik w postaci {label, liczebność dla danego labela}
    return results


def entropy(data):
    from math import log
    log2 = lambda x: log(x) / log(2)
    results = unique_labels_counter(data)

    entr = 0.0
    for r in results:
        p = float(results[r] ) /len(data)
        entr -= p * log2(p)
    return entr
