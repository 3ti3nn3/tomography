import qutip as qt

def main(argv = None):
    b = qt.Bloch()
    vec = [0, 0, 1]
    b.add_vectors(vec)
    b.save()
    return 0

if __name__ == '__main__':
    main()
