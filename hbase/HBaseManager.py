import happybase

class HBaseRow:

    def __init__(self, row_key, row_values, family_name=None):
        self.row_key = row_key
        if family_name:
            self.row_values = {family_name + ':' + k: v for k, v in row_values.items()}
        else:
            self.row_values = {k.replace(HBaseManager.FAMILY_NAME + ':', ''): v for k, v in row_values.items()}

    def __str__(self):
        return str((self.row_key, self.row_values))


class HBaseManager:
    HOST = '0.0.0.0'
    #HOST = 'hbase-docker'
    PORT = 9090
    BATCH_SIZE = 1000
    FAMILY_NAME = "data"

    def __init__(self, connection_pool: happybase.ConnectionPool):
        self.connection_pool = connection_pool

    def create_table(self, table_name: str, delete=True):

        with self.connection_pool.connection() as connection:
            tables = connection.tables()
            table_names = [table.decode("utf-8") for table in tables]
            if table_name not in table_names:
                print("### Creating table: {0} ###".format(table_name))
                connection.create_table(table_name, {self.FAMILY_NAME: dict()})
            elif delete:
                print("### Deleting and Creating table {0} ###".format(table_name))
                connection.delete_table(table_name, disable=True)
                connection.create_table(table_name, {self.FAMILY_NAME: dict()})
            else:
                print("### Table Exists: {0} ###".format(table_name))

    def batch_insert(self, table_name, batch_inserts):
        '''
        :type batch_inserts: list of HBaseRow
        :rtype: boolean
        '''
        with self.connection_pool.connection() as connection:
            table = connection.table(table_name)

            try:
                with table.batch(batch_size=self.BATCH_SIZE) as b:
                    for row in batch_inserts:
                        b.put(row.row_key, row.row_values)
            except ValueError:
                print("HBase Batch Insert Failed!")
                return False

            return True

    def batch_increment(self, table_name, batch_increments):
        '''
        :type batch_increments: list of HBaseRow
        :rtype: boolean
        '''
        with self.connection_pool.connection() as connection:
            table = connection.table(table_name)
            #test = table.counter_set(b'row-key', b'data:clo1"', 12)
            #test = table.counter_inc(b'row-key', "data:clo1")
            #test = table.counter_inc(b'row-key', "data:clo1")


            try:
                for row in batch_increments:
                    for col, value in row.row_values.items():
                        test = table.counter_inc(row.row_key, col, value = value)

            except ValueError:
                print("HBase Batch Insert Failed!")
                return False

            return True

    def batch_get_rows(self, table_name: str, row_keys: list):
        '''
        :rtype: list of HBaseRow
        '''
        with self.connection_pool.connection() as connection:
            table = connection.table(table_name)
            rows = table.rows(row_keys)
            hbase_rows = [HBaseRow(key.decode("utf-8"), self.convert_byte_to_utf(row_values)) for key, row_values in rows]
            return hbase_rows

    def convert_byte_to_utf(self, row_values: dict):
        converted_row = {k.decode("utf-8"): v.decode("utf-8") for k, v in row_values.items()}
        return converted_row

def test1():
    pool = happybase.ConnectionPool(size=3, host=HBaseManager.HOST, port=HBaseManager.PORT )
    hbase_manager = HBaseManager(pool)
    table_name = "test_table"
    insert_rows = [HBaseRow("key1", {"col1": "field1", "col2": "field2"}, family_name=HBaseManager.FAMILY_NAME),
                   HBaseRow("key2", {"col1": "field1", "col2": "field2"}, family_name=HBaseManager.FAMILY_NAME)]

    hbase_manager.create_table(table_name)
    hbase_manager.batch_insert(table_name, insert_rows)

    output = hbase_manager.batch_get_rows(table_name, ["key1", "key2"])
    print("output is: ")
    print(output[0])
    print(output[1])

def test2():
    pool = happybase.ConnectionPool(size=3, host=HBaseManager.HOST, port=HBaseManager.PORT )
    hbase_manager = HBaseManager(pool)
    table_name = "test_table"
    insert_rows = [HBaseRow("key1", {"col1": 1, "col2":2}, family_name=HBaseManager.FAMILY_NAME)]

    hbase_manager.create_table(table_name)
    hbase_manager.batch_increment(table_name, insert_rows)

    output = hbase_manager.batch_get_rows(table_name, ["key1", "key2"])
    print("output is: ")
    print(output[0])
    print(output[1])

def main():
    test2()


if __name__ == "__main__":
    main()
