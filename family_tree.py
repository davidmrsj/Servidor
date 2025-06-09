class FamilyTree:
    def __init__(self):
        self.children = {}
        self.parents = {}
        self.spouses = {}

    def add_person(self, name):
        self.children.setdefault(name, [])
        self.parents.setdefault(name, [])
        self.spouses.setdefault(name, None)

    def add_marriage(self, p1, p2):
        self.add_person(p1)
        self.add_person(p2)
        self.spouses[p1] = p2
        self.spouses[p2] = p1

    def add_child(self, parent1, parent2, child):
        for p in (parent1, parent2, child):
            self.add_person(p)
        self.children[parent1].append(child)
        self.children[parent2].append(child)
        self.parents[child].extend([parent1, parent2])

    def get_children(self, person):
        return self.children.get(person, [])

    def get_parents(self, person):
        return self.parents.get(person, [])

    def get_grandchildren(self, person):
        gcs = []
        for child in self.get_children(person):
            gcs.extend(self.get_children(child))
        return list(set(gcs))

    def get_grandparents(self, person):
        gps = []
        for parent in self.get_parents(person):
            gps.extend(self.get_parents(parent))
        return list(set(gps))

    def get_siblings(self, person):
        sibs = set()
        for parent in self.get_parents(person):
            for child in self.get_children(parent):
                if child != person:
                    sibs.add(child)
        return list(sibs)

    def get_cousins(self, person):
        cousins = set()
        for parent in self.get_parents(person):
            for sibling in self.get_siblings(parent):
                cousins.update(self.get_children(sibling))
        return list(cousins)

def create_sample_tree():
    tree = FamilyTree()
    # Level 1
    tree.add_marriage("Guillermo", "Carmen")
    tree.add_marriage("Tomas", "Julia")
    tree.add_marriage("Eduardo", "Elena")

    tree.add_child("Guillermo", "Carmen", "Carlos")
    tree.add_child("Tomas", "Julia", "Andres")
    tree.add_child("Eduardo", "Elena", "Manuel")

    # Level 2
    tree.add_marriage("Carlos", "Lucia")
    tree.add_marriage("Andres", "Sara")
    tree.add_marriage("Manuel", "Marta")

    tree.add_child("Carlos", "Lucia", "Daniel")
    tree.add_child("Andres", "Sara", "Javier")
    tree.add_child("Manuel", "Marta", "Roberto")

    # Level 3
    tree.add_marriage("Daniel", "Sofia")
    tree.add_marriage("Javier", "Laura")
    tree.add_marriage("Roberto", "Patricia")

    tree.add_child("Daniel", "Sofia", "Adrian")
    tree.add_child("Daniel", "Sofia", "Natalia")
    tree.add_child("Javier", "Laura", "Sergio")
    tree.add_child("Javier", "Laura", "Blanca")
    tree.add_child("Roberto", "Patricia", "Marcos")
    tree.add_child("Roberto", "Patricia", "Daniela")

    # Level 4
    tree.add_marriage("Adrian", "Paula")
    tree.add_marriage("Sergio", "Teresa")
    tree.add_marriage("Marcos", "Raquel")

    tree.add_child("Adrian", "Paula", "Mateo")
    tree.add_child("Adrian", "Paula", "Claudia")
    tree.add_child("Sergio", "Teresa", "Lucas")
    tree.add_child("Sergio", "Teresa", "Marina")
    tree.add_child("Marcos", "Raquel", "Hugo")
    tree.add_child("Marcos", "Raquel", "Ines")

    return tree

if __name__ == "__main__":
    tree = create_sample_tree()
    person = "Adrian"
    print(f"Nietos de {person}: {tree.get_grandchildren(person)}")
    person = "Mateo"
    print(f"Abuelos de {person}: {tree.get_grandparents(person)}")
    person = "Mateo"
    print(f"Hermanos de {person}: {tree.get_siblings(person)}")
    person = "Mateo"
    print(f"Primos de {person}: {tree.get_cousins(person)}")
