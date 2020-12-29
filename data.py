from tensorflow.keras.preprocessing.image import ImageDataGenerator
class DataGenerator :

    def train_data(self,path):
        train_generator = ImageDataGenerator(rescale=1./255,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             horizontal_flip=True
                                             )
        train_gen = train_generator.flow_from_directory(path,
                                                        class_mode='categorical',
                                                        batch_size=32,
                                                        target_size=(224,224))
        return train_gen

    def test_data(self,path):
        test_generator = ImageDataGenerator(rescale=1./255)
        test_gen = test_generator.flow_from_directory(path,
                                                        class_mode='categorical',
                                                        batch_size=32,
                                                        target_size=(150, 150))
        return test_gen

    def aug_data(self,image):
        #This function can be called from ImageDataGenerator for custom augumentation with preprocess_function
        pass