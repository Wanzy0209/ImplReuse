# pyright: strict
import tensorflow as tf
from tf.compat.v1.train import SessionCreator

# The original issue highlights that untyped implementations in core classes
# (like _NoParamDecoratorContextManager.__new__) obscure types.
# We test if SessionCreator's untyped create_session method causes similar issues.
class MySessionCreator(SessionCreator):
    def create_session(self):
        # The base class method is untyped, potentially obscuring the return type here.
        return None

creator = MySessionCreator()
reveal_type(creator.create_session)