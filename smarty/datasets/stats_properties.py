import numpy as np

from smarty.errors import assertion


class StatisticsProperty:
    """Statistics methods for a DataSet"""

    def fill_nan(self, key, strategy="mean"):
        """Fils empty cells in the cols according to choosen strategy

        :param str | list | int key: column name (str), list of column names (list of strs), column index (int) or list of columns indexes (list of ints)
        :param int | str strategy: mean - fill with mean value from column, median - same as mean, remove - removes rows with empty cells, othweriwse will put value of strategy to empty cells
        :raises: AssertionError if key is not valid
        :returns: DataSet of dropped rows (if any)
        """
        if isinstance(key, str): # single column by name
            key = [self._get_idx(key)]
        elif isinstance(key, list) and isinstance(key[0], str): # multiple columns by names
            key = [self._get_idx(k) for k in key]
        elif isinstance(key, int):
            key = [key]
        assertion(type(key) == int or (type(key) == list and type(key[0]) == int) or type(key) == slice, "Wrong column specifier")

        remove_idxs = set()
        for idx in key:
            if self._is_categorical(idx):
                nan = np.where(self.matrix_[:, idx] == "nan")[0]
            else:
                nan = np.where(np.isnan(self.matrix_[:, idx].astype("f")))
    
            if strategy == "mean":
                fill_val = np.nanmean(self.matrix_[:, idx])
            elif strategy == "median":
                fill_val = np.nanmedian(self.matrix_[:, idx])
            elif strategy == "remove":
                remove_idxs.update(nan)
                continue
            else:
                fill_val = strategy
            self[nan, idx] = fill_val # to check dtype

        remove_idxs = [int(x) for x in remove_idxs]
        if len(remove_idxs) != 0:
            return self.drop_r(remove_idxs)
            
    def info(self):
        """Displays basic information about the dataset, like number of rows, columns, columns names and dtypes"""

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

    def head(self, num=5): 
        """Prints up to num first row of matrix\_, or if dataset is empty prints 'DataSet is empty'
        
        :param int num: how many rows to print
        """

        if self.empty_():
            print("DataSet is empty.")
            return

        num = min(self.get_shape_()[0], num)
        print()
        print(f"First {num} rows of DataSet at {hex(id(self))}:")
        print(" " * 8 + "".join(f"{name : <15}  " for name in self.columns_))
        for idx in range(num):
            print(f"{idx : >6}  " + "".join(f"{val :<15}  " for val in self.matrix_[idx, :]))
        print()

    def descr_num(self):
        """Prints statistical data about numerical columns of the dataset (if it is not empty, else prints 'DataSet is empty'), like 

        1. count of non-nan entires
        2. mean value
        3. standard deviation
        4. minimum value, 25, 50, 75th percentile, maximum value
        """

        if self.empty_():
            print("DataSet is empty.")
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

    def descr_cat(self, k=3):
        """Prints statistical data about categorical columns of the dataset (if it is not empty, else prints 'DataSet is empty'), like 

        1. count of non-nan entires
        2. number of unique entries (empty entry is also counted as type of entry)
        3. entries frequency
        4. up to k most frequent items for each column

        :param int k: number of unique entries to display
        """

        if self.empty_():
            print("DataSet is empty.")
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

    def _not_nan_cat(self): # categorical
        return [len(np.where(self.matrix_[:, idx] != "nan")[0]) for idx in self.categorical_idxs_()]

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
        all_ = [np.unique(self.matrix_[:, idx], return_counts=True) for idx in self.categorical_idxs_()]
        all_ = [sorted(list(zip(*a)), key=lambda x: x[1], reverse=True) for a in all_]
        return all_
     

