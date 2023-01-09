from lenses import lens


class IDict(dict):

    def __hash__(self):
        return id(self)

    def _immutable(self, *args, **kws):
        raise TypeError('Object is immutable')

    def pop(self, k):
        return lens[k].collect()

    def set(self, k, v):
        return lens[k].set(v)(dict(self))

    __delitem__ = _immutable
    __setitem__ = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable
    popitem = _immutable


class IList(list):

    def __hash__(self):
        return id(self)

    def _immutable(self, *args, **kws):
        raise TypeError('Object is immutable')

    __setitem__ = _immutable
    __delitem__ = _immutable
    pop = _immutable
    insert = _immutable
    push = _immutable
    append = _immutable
    extend = _immutable


if __name__ == "__main__":
    d = IDict(x=1, y=2)
    print(d)
    d = d.set('x', 2)
    print(d)
