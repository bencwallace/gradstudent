import gdb
import gdb.printing


class ArrayPrinter:
    def __init__(self, val):
        self.__val = val
        self.data = self.__val["data_"]
        self.pp = gdb.default_visualizer(self.data)
        self._children = list(self.pp.children())

    def _iter(self):
        for i in range(int(self.__val["size_"])):
            yield ("[%d]" % i, self._children[0][1][i])

    def children(self):
        return self._iter()

    def to_string(self):
        elems = ", ".join(str(x[1]) for x in self._iter())
        return f"({elems})"


def build_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter("gradstudent")
    pp.add_printer("Array", "^gs::Array$", ArrayPrinter)
    return pp


gdb.printing.register_pretty_printer(
    gdb.current_objfile(),
    build_pretty_printer(),
)
