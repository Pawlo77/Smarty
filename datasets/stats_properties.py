import numpy as np


class StatisticsProperty:
    # number of rows and columns, column names and dtypes
    def info(self):
        print()
        if self.empty_():
            print(f"Empty DataMatrix at {hex(id(self))}")
            print()
            return

        print(f"DataSet at {hex(id(self))}")
        print(f"\tRows: {self.get_shape_()[0]}")
        print(f"\tCols: {self.get_shape_()[1]}")
        print()
        print(f"\t{'name' : <15}\t{'dtype' : >15}")
        for name, dtype in zip(self.columns_, self.dtypes_):
            print(f"\t{name : <15}\t{str(dtype) : >15}")
        print()

    # print up to first num rows of matrix_
    def head(self, num=5): 
        if self.empty_():
            print("DataMatrix is empty.")
            return

        num = min(self.get_shape_()[0], num)
        print()
        print(f"First {num} rows of DataSet at {hex(id(self))}:")
        print(" " * 8 + "".join(f"{name : <15}  " for name in self.columns_))
        for idx in range(num):
            print(f"{idx : >6}  " + "".join(f"{val :<15}  " for val in self.matrix_[idx, :]))
        print()

    # prints statistical data about numerical columns
    def descr_num(self):
        if self.empty_():
            print("DataMatrix is empty.")
            return

        print(" " * 10 + "".join(f"{self.columns_[idx] : <15}  " for idx in self.numerical_idxs_()))

        info = (
            ("count", self._not_nan_num()),
            ("mean", self._mean()),
            ("std_dev", self._std()),
            ("min", self._min()),
            ("25%", self._25th()),
            ("50%", self._50th()),
            ("75%", self._75th()),
            ("max", self._max()),
        )

        for name, vals in info:
            print(f"{name : >8}  " + "".join(f"{val : <15.5f}  " for val in vals))
        print("   dtype  "  + "".join(f"{str(self.dtypes_[idx]) : <15}  " for idx in self.numerical_idxs_()))

    # prints statistical data about categorical columns
    # k - k-most frequent items for each column
    def descr_cat(self, k=6):
        if self.empty_():
            print("DataMatrix is empty.")
            return

        print(" " * 10 + "".join(f"{self.columns_[idx] : <15}  " for idx in self.categorical_idxs_()))

        not_nan = self._not_nan_cat()
        unique = self._unique_num()

        unique_num = [len(u) for u in unique]
        freq = np.array(unique_num) / np.array(not_nan)

        for i in range(len(unique)):
            if len(unique[i]) >= k:
                unique[i] = unique[i][:k]
            else:
                unique[i] += [("-", "")] * (k - len(unique[i]))

        info = (
            ("count", not_nan),
            ("unique", unique_num),
            ("freq", freq),
        )
        
        for name, vals in info:
            print(f"{name : >8}  " + "".join(f"{val : <15.5f}  " for val in vals))
        
        for i in range(k):
            # it k exceedes number of unique elements, print -
            print(f"{'Top' + str(i+1) : >8}  " + "".join([f"{val[i][0] + ' (' + str(val[i][1]) + ')' : <15}  " if val[i][1] != "" else f"{'-' : ^15}  " for val in unique]))

        print("   dtype  "  + "".join(f"{str(self.dtypes_[idx]) : <15}  " for idx in self.categorical_idxs_()))

    def _not_nan_num(self): # numeric
        return [np.count_nonzero(~np.isnan(self.matrix_[:, idx].astype("f"))) for idx in self.numerical_idxs_()]

    def _not_nan_cat(self): # categorical, fix here
        return [np.count_nonzero(self.matrix_[:, idx].astype("O")) for idx in self.categorical_idxs_()]

    def _mean(self): # numeric
        return [np.nanmean(self.matrix_[:, idx].astype("f"))for idx in self.numerical_idxs_()]

    def _std(self): # numeric
        return [np.nanstd(self.matrix_[:, idx].astype("f"))for idx in self.numerical_idxs_()]

    def _min(self): # numeric
        return [np.nanmin(self.matrix_[:, idx].astype("f"))for idx in self.numerical_idxs_()]

    def _25th(self): # numeric
        return [np.nanpercentile(self.matrix_[:, idx].astype("f"), 25)for idx in self.numerical_idxs_()]
        
    def _50th(self): # numeric
        return [np.nanpercentile(self.matrix_[:, idx].astype("f"), 50)for idx in self.numerical_idxs_()]

    def _75th(self): # numeric
        return [np.nanpercentile(self.matrix_[:, idx].astype("f"), 75)for idx in self.numerical_idxs_()]

    def _max(self): # numeric
        return [np.nanmax(self.matrix_[:, idx].astype("f"))for idx in self.numerical_idxs_()]

    # returns a 3 d array -> each column -> each unique item -> [item, frequency]
    def _unique_num(self): # categorical
        all_ = [np.unique(self.matrix_[:, idx].astype("U"), return_counts=True) for idx in self.categorical_idxs_()]
        all_ = [sorted(list(zip(*a)), key=lambda x: x[1]) for a in all_]
        return all_
     

