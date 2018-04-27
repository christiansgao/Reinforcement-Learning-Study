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
    PORT = 9090
    BATCH_SIZE = 1000
    FAMILY_NAME = "data"

    def __init__(self):
        self.connection = happybase.Connection(self.HOST, self.PORT)

    def create_table(self, table_name: str, delete=True):
        tables = self.connection.tables()
        table_names = [table.decode("utf-8") for table in tables]
        if table_name not in table_names:
            print("### Creating table: {0} ###".format(table_name))
            self.connection.create_table(table_name, {self.FAMILY_NAME: dict()})
        elif delete:
            print("### Deleting and Creating table {0} ###".format(table_name))
            self.connection.delete_table(table_name, disable=True)
            self.connection.create_table(table_name, {self.FAMILY_NAME: dict()})
        else:
            print("### Table Exists: {0} ###".format(table_name))

    def batch_insert(self, table_name, batch_inserts):
        '''
        :type batch_inserts: list of HBaseRow
        :rtype: boolean
        '''

        table = self.connection.table(table_name)

        try:
            with table.batch(batch_size=self.BATCH_SIZE) as b:
                for row in batch_inserts:
                    b.put(row.row_key, row.row_values)
        except ValueError:
            print("HBase Batch Insert Failed!")
            return False

        return True

    def batch_get_rows(self, table_name: str, row_keys: list):
        '''
        :rtype: list of HBaseRow
        '''

        table = self.connection.table(table_name)
        rows = table.rows(row_keys)
        hbase_rows = [HBaseRow(key.decode("utf-8"), self.convert_byte_to_utf(row_values)) for key, row_values in rows]
        return hbase_rows

    def convert_byte_to_utf(self, row_values: dict):
        converted_row = {k.decode("utf-8"): v.decode("utf-8") for k, v in row_values.items()}
        return converted_row


def main():
    hbase_manager = HBaseManager()
    table_name = "test_table"
    insert_rows = [HBaseRow("key1", {"col1": "field1", "col2": "field2"}, family_name=HBaseManager.FAMILY_NAME),
                   HBaseRow("key2", {"col1": "field1", "col2": "field2"}, family_name=HBaseManager.FAMILY_NAME)]

    hbase_manager.create_table(table_name)
    hbase_manager.batch_insert(table_name, insert_rows)

    output = hbase_manager.batch_get_rows(table_name, ["key1", "key2"])
    print("output is: ")
    print(output[0])
    print(output[1])


if __name__ == "__main__":
    main()
