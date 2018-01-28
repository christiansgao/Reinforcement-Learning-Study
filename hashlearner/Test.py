def foo(bar, baz):
  print('hello {0}'.format(bar))
  return 'foo' + baz, "foo"

from multiprocessing.pool import ThreadPool
pool = ThreadPool(processes=8)

for i in range(10):
    print(i)
    async_result = pool.apply_async(foo, ('world', 'foo')) # tuple of args for foo

# do some other stuff in the main process

return_val = async_result.get()