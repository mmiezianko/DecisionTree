
import collections
from algorithm import *
import itertools

class DecisionTree:
    id_iter = itertools.count()
    """Klasa drzewo decyzyjne. """
    def __init__(self, col=-1, value=None, branch_with_value=None, branch_with_others=None, outputs=None, columns_map=None, size = 0):
        self.id = next(DecisionTree.id_iter)
        self.branch_with_value = branch_with_value
        self.branch_with_others = branch_with_others
        self.value = value
        self.col = col # kolumny
        self.outputs = outputs # None dla wezłow, not None dla liści
        if columns_map is not None:
            self.col_name = columns_map[self.col]
        else:
            self.col_name = str(self.col)
        self.set_size = size

def grow_tree(data, algorithm_fun = entropy, columns_map=None):
    """Implementacja algorytmu ID3"""
    size = len(data)
    # data -> rekordy tabeli; jeżeli są = 0 to zwracamy puste drzewo
    if size == 0: return DecisionTree()
    # algorithm_fun dla zbioru danych w pierwszej interacji bedzie sie odnosił do entropii całego zbioru danych.
    # w kolenych iteracjach będzie to entropia poprzedniej decyzji
    current_result = algorithm_fun(data) # obliczanie entropii ukadu

    # best gain to najwyzszy wskaznik jakosci
    best_info_gain = 0.0
    best_value_labelled = None
    best_subsets = None

    col_num = len(data[0]) - 1  # zliczanie ilości zmiennych opisujących. ostatnia kolumna to target dlatego - 1
    for col in range(col_num):
        # iterowanie po zmiennych opisujących. zmienna values_of_column będzie zawierać listę wszystkich zmiennych dla danej kolumny
        # czyli dla ostatniej kolumny [zmeczony, srednie, wypoczety, srednie, srednie ... itd ]
        values_of_column = [row[col] for row in data]

        for value in values_of_column:
            # wyciągamy unikalne wartości (np. kobieta/mężczyzna, przedziały wiekowe itd.)
            # subset1, subset2 to są podzbiory: w jednym jest dana unikalna wartość,
            # w drugim znajduje się pozostała reszta unikalnych wartości
            (subset1, subset2) = set_splitter(data, col, value)

            # p to prawdopodobieństwo wystąpienia danej wartości
            p = float(len(subset1)) / len(data)
            #wyliczenie zysku  -> szukamy max
            info_gain = current_result - p*algorithm_fun(subset1) - (1-p)*algorithm_fun(subset2)
            #jeśli podzbiory nie są puste (czyli mamy z czego podział) i info_gain jest większy od best_info_gain to:
            if info_gain > best_info_gain and len(subset1)>0 and len(subset2) > 0:
                best_info_gain = info_gain
                best_value_labelled = (col, value) #col to nazwa zmiennej decyzyjnej value to jej wartość np. płeć, kobieta
                best_subsets = (subset1, subset2)

    # jeżeli wystąpił zysk to dzielimy dalej -> powtarzamy proces
    if best_info_gain > 0: # gdyby dać tu 0.5 to byłby prepruning
        #rekurencja
        branch_with_value = grow_tree(best_subsets[0], columns_map=columns_map) # gałąź ktora zawiera daną cechę (bardziej informatywną)
        branch_with_others =grow_tree(best_subsets[1], columns_map=columns_map) # gałąź z resztą cech
        # branch_with_value i branch_with_others zapewniają rekurencję, czyli "zadajemy" kolejne pytania, tak długo az zysk informacyjny = 0
        # depth drzewa rozwija się właśnie na tym etapie. W sytuacji gdy zysk informacyjny = 0, uruchamiany jest else, w którym obliczane są
        # wystąpienia danej etykiety w liściu
        # depth -> czyli głębokość drzewa to kolejne instancje klasy Decision Tree - węzły
        return DecisionTree(col=best_value_labelled[0], value=best_value_labelled[1], branch_with_value=branch_with_value, branch_with_others=branch_with_others,columns_map=columns_map, size = size)
    else:
        # zwraca liczebnośći labeli w formie słownika np. skały: 10, dom: 9, ścianka: 5
        return DecisionTree(outputs=unique_labels_counter(data), size = size)
    # DecisionTree zawiera wskaźniki na instancje klasy DecisionTree ( te wskaźniki to gałęxie - "małe drzewa")

# jest to tzw. post pruning. Robimy to aby zrozumieć lepiej działanie drzewa, ale równie dobrze można by zrobić
# pre pruning czyli obcinanie już na poziomie tworzenia drzewa ustwiając if best_info_gain > 0.5:
def prune(tree, minGain, algorithm_fun=entropy, notification=False): # tree to jest węzeł i dwa podproblemy. Drzewo zawiera w
    """Pruning według minimalnego zysku informacyjnego. """
    
    # wywołujemy funkcje prune rekurencyjnie dla każdego brancha - dziecka (jest nim drzewo, argument tree)
    
    # Jeśli output gałązi z wartością jest pusty (czyli jeśli nie jest to liść)  
    if tree.branch_with_value.outputs == None: 
        prune(tree.branch_with_value, minGain, algorithm_fun, notification) # przechodzimy do koljenego poziomu

    # Jeśli output gałązi z resztą wartości jest pusty (czyli jeśli nie jest to liść)  
    if tree.branch_with_others.outputs == None: 
        prune(tree.branch_with_others, minGain, algorithm_fun, notification) # przechodzimy do koljenego poziomu

    
    # jeśli jesteśmy już w liściu:
    
    if tree.branch_with_value.outputs != None and tree.branch_with_others.outputs != None:
        output_of_branch_with_value, output_of_branch_with_others = [], []  #

        # tutaj iterujemy po key oraz po value outputs (czyli np. skały:2, dom:3, ściana:4) i otrzymujemy listę
        # gdzie jest tyle wsytąpień labela, co wartość w słowniku (czyli np. skały, skały, dom, dom, dom)
        for v, c in tree.branch_with_value.outputs.items(): # items odnosi się do key i value w słowniku, czyli label wraz z liczebnością
            for i in range(c):
                output_of_branch_with_value.append(v)
        for v, c in tree.branch_with_others.outputs.items():
            for i in range(c):
                output_of_branch_with_others.append(v)

        # powtarzamy operacje z funkcji grow_tree

        # p to prawdopodobieństwo wystąpienia danej wartości
        p = float(len(output_of_branch_with_value)) / len(output_of_branch_with_value + output_of_branch_with_others)
        # wyliczenie zysku  -> szukamy max. feature_importance to inaczej zysk informacyjny, który wskazuje na to czy liście powinny zostać obcięte (czy podział na gałęzie niesie zysk informacyjny)
        feature_importance = algorithm_fun(output_of_branch_with_value + output_of_branch_with_others) - p*algorithm_fun(output_of_branch_with_value) - (1-p)*algorithm_fun( output_of_branch_with_others)
        if feature_importance < minGain: # minGain to nasz treshold
            if notification: print('Nastąpił pruning: zysk informacyjny = %f' % feature_importance)
            tree.branch_with_value, tree.branch_with_others = None, None # ucięcie liści
            tree.outputs = unique_labels_counter(output_of_branch_with_value + output_of_branch_with_others) # zawiązanie liścia w miejsce następnika


def predict(samples, tree_model, dataMissing=False):
  """Klasyfikacja obserwacji zgodnie z danym drzewem """

  # w tej funkcji sprawdzam które prawdopodobieństwo z outputs było największe
  # czyli np. jak mamy w outputs 2/10 kobiet w wieku > 30 lat mieszkających  na wsi = internet, 6/10 ... = prasa, 2/10 telewizja
  # to zaklasyfikuje nam, że  kobiety w wieku > 30 lat mieszkające  na wsi czytają prasę
  # zaczynamy od początku drzewa
  if tree_model.outputs != None:  # liść
      #sprawdza czy skończyliśmy szukanie 
      value_counts = 0
      decision = {'predykcja':None,'liczba':0}
      # iterujemy po kluczu i vartosci w slowniku outputow
      for k,v in tree_model.outputs.items():
        # jezeli liczebnosc danej kategorii jest wieksza od poprzedniej to nadpisujemy slownik
        if v>=decision['liczba']:
          decision['predykcja']=k
          decision['liczba']=v
        # zliczamy wszystkie liczebnosci etykiety dla danego liscia
        value_counts += v
      decision['dokladnosc'] = decision['liczba']/value_counts

      return decision
  else:
      #dla węzła pobieramy wartość z sample która odpowiada atrybutowi decyzyjnemu w drzewie
      v = samples[tree_model.col] # col=best_value_labelled[0] czyli label
      branch = None # referencja na kolejne drzewo
      if isinstance(v, int) or isinstance(v, float):
          if v >= tree_model.value:
              branch = tree_model.branch_with_value
          else:
              branch = tree_model.branch_with_others
      else:
          # sprawdza, czy pobrana wartość jest równa true branch (gałąź z wartością) czy false branch (gałąź z pozostałymi wartościami) 
          if v == tree_model.value:
              branch = tree_model.branch_with_value # referencja na gałąź
          else:
              branch = tree_model.branch_with_others # referencja na gałąź
  # wywoułujemy rekurencyjnie (a za każdym razem na początku funkcji sprawdzamy czy doszliśmy już do liścia)
  return predict(samples, branch)

def plot(tree):

    def print_node(tree, indent=''):
        if tree.outputs != None:  # leaf node
            return str(tree.outputs)
        else:
            if isinstance(tree.value, int) or isinstance(tree.value, float):
                decision = f'id={tree.__hash__()} Column {tree.col_name}: x >= {tree.value}?'
            else:
                decision = f'id={tree.__hash__()},Column {tree.col_name}: x == {tree.value}?'
            branch_with_value = indent + 'yes -> ' + print_node(tree.branch_with_value, indent + '\t\t')
            branch_with_others = indent + 'no  -> ' + print_node(tree.branch_with_others, indent + '\t\t')
            return (decision + '\n' + branch_with_value + '\n' + branch_with_others)

    print(print_node(tree))


