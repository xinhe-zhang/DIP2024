from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
#-------------------------------------------------------

# Q0-1
def r_img(input_path, output_path):
    img = imread(input_path)
    flipped = img[::-1, :, :]
    imsave(output_path, flipped)

# Q0-2
def greyscale(input_path, output_path):
    img = imread(input_path)
    gray_img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    imsave(output_path, gray_img, cmap='gray')


# Q1-a
def debright(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    new_img = img / 3
    cv2.imwrite(output_path, new_img)

# Q1-b
def inbright(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    new_img = img * 3
    cv2.imwrite(output_path, new_img)

# Q1-c
#plots below

# Q1-d
def GHE(input_path, output_path):
    img = imread(input_path)
    shape = img.shape
    img = (img * 255).astype(np.uint8)
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
    pdf = hist / (hist.sum())
    cdf = pdf.cumsum()
    img_value = img.flatten()

    img_value = (cdf[img_value]* 255).astype(np.uint8)
    img_value = img_value/255
    img_value = np.array(img_value)
    mapped_img = img_value.reshape(shape)
    imsave(output_path, mapped_img, cmap='gray')

#plots below
    
# Q1-e
def LHE(input_path, output_path, kernel_size):
    img = np.array(Image.open(input_path))

    def my_pad(img, pad_size):
        padded_img = np.zeros((img.shape[0] + 2 * pad_size, img.shape[1] + 2 * pad_size), dtype=img.dtype)
        padded_img[pad_size:-pad_size, pad_size:-pad_size] = img
        padded_img[:pad_size, pad_size:-pad_size] = img[0:pad_size, :]
        padded_img[-pad_size:, pad_size:-pad_size] = img[-pad_size:, :]
        return padded_img

    def compute_pdf(slice_i):
        hist, _ = np.histogram(slice_i, bins=np.arange(257), density=True)
        return hist

    def compute_cdf(pdf):
        cdf = np.cumsum(pdf)
        return cdf

    def apply_equalization(slice_i, cdf):
        center = int(slice_i[1, 1])  
        new_center = np.round(cdf[center] * 255).astype(np.uint8) #transform to uniform distribution  
        return new_center

    def reshape_matrix(matrix, shape):
        return matrix.reshape(shape)

    pad_size = kernel_size[0] // 2
    padded_img = my_pad(img, pad_size)

    slices = [padded_img[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1] for i in range(pad_size, img.shape[0]+pad_size) for j in range(pad_size, img.shape[1]+pad_size)]
    cdfs = [compute_cdf(compute_pdf(slice_i)) for slice_i in slices]

    equalized_slices = [apply_equalization(slice_i, cdf) for slice_i, cdf in zip(slices, cdfs)]
    equalized_image = reshape_matrix(np.array(equalized_slices), img.shape)

    equalized_image = (equalized_image * 255).astype(np.uint8)

    Image.fromarray(equalized_image).save(output_path)

#plots below
    
# Q1-f
def blackcat(input_path, output_path, t, s):
    img = imread(input_path)
    shape = img.shape
    img = (img * 255).astype(np.uint8)
    hist, __ = np.histogram(img.flatten(), bins=256, range=[0, 256])
    pdf = hist / (hist.sum())
    cdf = pdf.cumsum()
    img_value = img.flatten()

    for idx, value in enumerate(img_value):
        if value <= t: #dark
            img_value[idx] = (cdf[value] * s * 255).astype(np.uint8)  
        else: #bright
            img_value[idx] = (cdf[value] * 255).astype(np.uint8)  

    mapped_img = img_value / 255
    mapped_img = np.array(mapped_img)
    mapped_img = mapped_img.reshape(shape)
    imsave(output_path, mapped_img, cmap='gray')

#plots below
    
#Q2-a
def gaussian_noise_bye(input_path, output_path, kernel_size):
    img = imread(input_path)
    shape = img.shape

    def my_pad(img, pad_size):
        if len(img.shape) == 3:  
            img_gray = np.mean(img, axis=2)  
        else:
            img_gray = img  

        padded_img = np.zeros((img_gray.shape[0] + 2 * pad_size, img_gray.shape[1] + 2 * pad_size), dtype=img_gray.dtype)
        padded_img[pad_size:-pad_size, pad_size:-pad_size] = img_gray
        padded_img[:pad_size, pad_size:-pad_size] = img_gray[0:pad_size, :]
        padded_img[-pad_size:, pad_size:-pad_size] = img_gray[-pad_size:, :]
        return padded_img
    
    def weighted_mean(slice_i):
        n = len(slice_i)
        #select center
        total_weighted_sum  = 2 * slice_i[n // 2][n // 2]  
        total_weight = 2  
        #select the others
        for i in range(n):
            for j in range(n):
                if i != n // 2 or j != n // 2: #exclude center value
                    total_weighted_sum += slice_i[i][j]
                    total_weight += 1 
        weighted_mean = total_weighted_sum / total_weight
        return weighted_mean
    
    def gaus_noise_filter(slice_i):
        new_middle = weighted_mean(slice_i)
        return new_middle

    def reshape_matrix(matrix, shape):
        return matrix.reshape(shape)

    pad_size = kernel_size[0] // 2
    padded_img = my_pad(img, pad_size)

    slices = [padded_img[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1] for i in range(pad_size, shape[0]+pad_size) for j in range(pad_size, shape[1]+pad_size)]
    
    clean_slices = [gaus_noise_filter(slice_i) for slice_i in slices]
    clean_slices = reshape_matrix(np.array(clean_slices), shape[0:2])
    clean_image = (clean_slices * 255).astype(np.uint8)

    Image.fromarray(clean_image).save(output_path)


#Q2-a
def saltpepper(input_path, output_path, kernel_size):
    img = imread(input_path)
    shape = img.shape

    def my_pad(img, pad_size):
        padded_img = np.zeros((img.shape[0] + 2 * pad_size, img.shape[1] + 2 * pad_size, img.shape[2]), dtype=img.dtype)
        padded_img[pad_size:-pad_size, pad_size:-pad_size] = img
        padded_img[:pad_size, pad_size:-pad_size] = img[0:pad_size, :]
        padded_img[-pad_size:, pad_size:-pad_size] = img[-pad_size:, :]
        return padded_img

    def MAXMIN(num):
        min_values = []
        for i in range(len(num) - 2):
            min_value = np.min(num[i:i+3])
            min_values.append(min_value)
        max_result = np.max(min_values)
        return max_result

    def MINMAX(num):
        max_values = []
        for i in range(len(num) - 2):
            max_value = np.max(num[i:i+3])
            max_values.append(max_value)
        min_result = np.min(max_values)
        return min_result

    def PMED(slice_i):
        maxmin_result = MAXMIN(slice_i)
        minmax_result = MINMAX(slice_i)
        result = (maxmin_result + minmax_result) / 2
        return result

    def reshape_matrix(matrix, shape):
        return matrix.reshape(shape)

    pad_size = kernel_size[0] // 2
    padded_img = my_pad(img, pad_size)
   
    row_slices = [padded_img[i, j-pad_size:j+pad_size+1] for i in range(pad_size, shape[0]+pad_size) for j in range(pad_size, shape[1]+pad_size)]
    col_slices = [padded_img[i-pad_size:i+pad_size+1, j] for i in range(pad_size, shape[0]+pad_size) for j in range(pad_size, shape[1]+pad_size)]
    

    clean_row_slices = [PMED(slice_i) for slice_i in row_slices]
    clean_col_slices = [PMED(slice_i) for slice_i in col_slices]
    clean_slices = (np.array(clean_row_slices) + np.array(clean_col_slices)) / 2
    clean_image = reshape_matrix(np.array(clean_slices), img.shape[0:2])
    clean_image = (clean_image * 255).astype(np.uint8)

    imsave(output_path, clean_image, cmap='gray')

#Q2-b
def PSNR(original, filtered):
    ori_img = imread(original)
    ori_img = ori_img*255
    ori_img = ori_img.flatten()
    filt_img = imread(filtered)
    filt_img = filt_img*255
    filt_img = filt_img.flatten()
    def MSE(original, filtered):
        SE = []
        for i in range(len(original)):
            SE_i = np.square(original[i] - filtered[i])
            SE.append(SE_i)   
        result = np.mean(SE)
        return result
    MSE_value = MSE(ori_img, filt_img)
    PSNR_value = round(10 * np.log10(np.square(255)/MSE_value), 2)
    print(filtered, "PSNR :", PSNR_value, "dB") 

#Q2-c
def mix_noise(input_path, output_path, kernel_size1, kernel_size2):
    saltpepper(input_path, output_path, kernel_size1)
    def gaussian_noise_bye2(input_path, output_path, kernel_size, repeat):

        def my_pad(img, pad_size):
            if len(img.shape) == 3:  
                img_gray = np.mean(img, axis=2)  
            else:
                img_gray = img  

            padded_img = np.zeros((img_gray.shape[0] + 2 * pad_size, img_gray.shape[1] + 2 * pad_size), dtype=img_gray.dtype)
            padded_img[pad_size:-pad_size, pad_size:-pad_size] = img_gray
            padded_img[:pad_size, pad_size:-pad_size] = img_gray[0:pad_size, :]
            padded_img[-pad_size:, pad_size:-pad_size] = img_gray[-pad_size:, :]
            return padded_img
        
        def weighted_mean(slice_i):
            n = len(slice_i)
            #select center
            total_weighted_sum  = 2 * slice_i[n // 2][n // 2]  
            total_weight = 2  
            #select the others
            for i in range(n):
                for j in range(n):
                    if i != n // 2 or j != n // 2: #exclude center value
                        total_weighted_sum += slice_i[i][j]
                        total_weight += 1 
            weighted_mean = total_weighted_sum / total_weight
            return weighted_mean
        
        def gaus_noise_filter(slice_i):
            new_middle = weighted_mean(slice_i)
            return new_middle

        def reshape_matrix(matrix, shape):
            return matrix.reshape(shape)
        
        for i in range(repeat):
            img = imread(input_path)
            shape = img.shape

            pad_size = kernel_size[0] // 2
            padded_img = my_pad(img, pad_size)

            slices = [padded_img[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1] for i in range(pad_size, shape[0]+pad_size) for j in range(pad_size, shape[1]+pad_size)]
            
            clean_slices = [gaus_noise_filter(slice_i) for slice_i in slices]
            clean_slices = reshape_matrix(np.array(clean_slices), shape[0:2])
            clean_image = (clean_slices * 255).astype(np.uint8)

            Image.fromarray(clean_image).save(output_path)
    gaussian_noise_bye2(output_path, output_path, kernel_size2, repeat=5)

    
# run main
if __name__ == "__main__":

    #file paths
    psample1 = "./SampleImage/sample1.png"
    psample2 = "./SampleImage/sample2.png"
    psample3 = "./SampleImage/sample3.png"
    psample4 = "./SampleImage/sample4.png"
    psample5 = "./SampleImage/sample5.png"
    psample6 = "./SampleImage/sample6.png"
    psample7 = "./SampleImage/sample7.png"
    presult1 = "./result1.png"
    presult2 = "./result2.png"
    presult3 = "./result3.png"
    presult4 = "./result4.png"
    presult5 = "./result5.png"
    presult6 = "./result6.png"
    presult7 = "./result7.png"
    presult8 = "./result8.png"
    presult9 = "./result9.png"
    presult10 = "./result10.png"
    presult11 = "./result11.png"
    presult12 = "./result12.png"

    output_sample2_hist = './plot/sample2_hist.png'
    output_sample3_hist = './plot/sample3_hist.png'
    output_result3_hist = './plot/result3_hist.png'
    output_result4_hist = './plot/result4_hist.png'
    output_result5_hist = './plot/result5_hist.png'
    output_result6_hist = './plot/result6_hist.png'
    output_result7_hist = './plot/result7_hist.png'
    output_result8_hist = './plot/result8_hist.png'
    output_result9_hist = './plot/result9_hist.png'

    #run funtions
    r_img(psample1, presult1)
    print("---result1 done!---")
    greyscale(presult1, presult2)
    print("---result2 done!---")
    debright(psample2, presult3)
    print("---result3 done!---")
    inbright(presult3, presult4)
    print("---result4 done!---")
    GHE(psample2, presult5)
    print("---result5 done!---")
    GHE(presult3, presult6)
    print("---result6 done!---")
    GHE(presult4, presult7)
    print("---result7 done!---")
    LHE(psample2, presult8, kernel_size=(7, 7))
    print("---result8 done!---")
    blackcat(psample3, presult9, 20, 0.78)
    print("---result9 done!---")
    gaussian_noise_bye(psample5, presult10, kernel_size=(9, 9))
    print("---result10 done!---")
    saltpepper(psample6, presult11, kernel_size=(9, 9))
    print("---result11 done!---")
    PSNR(psample4, presult10)
    PSNR(psample4, presult11)
    mix_noise(psample7, presult12, kernel_size1=(9, 9), kernel_size2=(3, 3))
    print("---result12 done!---")
    PSNR(psample4, presult12)

    #plotting
    '''
    sample2 = cv2.imread(psample2, cv2.IMREAD_GRAYSCALE)
    plt.hist(sample2.flatten(), bins=256, range=(0, 256), density=True)
    plt.title("sample2")
    plt.xlabel("grayscale")
    plt.ylabel("density")
    plt.savefig(output_sample2_hist)
    plt.show();

    result3 = cv2.imread(presult3, cv2.IMREAD_GRAYSCALE)
    plt.hist(result3.flatten(), bins=256, range=(0, 256), density=False)
    plt.title("result3")
    plt.xlabel("grayscale")
    plt.ylabel("density")
    plt.savefig(output_result3_hist)
    plt.show();

    result4 = cv2.imread(presult4, cv2.IMREAD_GRAYSCALE)
    plt.hist(result4.flatten(), bins=256, range=(0, 256), density=False)
    plt.title("result4")
    plt.xlabel("grayscale")
    plt.ylabel("density")
    plt.savefig(output_result4_hist)
    plt.show();

    sample5 = cv2.imread(presult5)
    plt.hist(sample5.flatten(), bins=256, range=(0, 256), density=False)
    plt.title("result5")
    plt.xlabel("grayscale")
    plt.ylabel("density")
    plt.savefig(output_result5_hist)
    plt.show();

    result6 = cv2.imread(presult6, cv2.IMREAD_GRAYSCALE)
    plt.hist(result6.flatten(), bins=256, range=(0, 256), density=False)
    plt.title("result6")
    plt.xlabel("grayscale")
    plt.ylabel("density")
    plt.savefig(output_result6_hist)
    plt.show();

    result7 = cv2.imread(presult7, cv2.IMREAD_GRAYSCALE)
    plt.hist(result7.flatten(), bins=256, range=(0, 256), density=False)
    plt.title("result7")
    plt.xlabel("grayscale")
    plt.ylabel("density")
    plt.savefig(output_result7_hist)
    plt.show();

    result8 = cv2.imread(presult8, cv2.IMREAD_GRAYSCALE)
    plt.hist(result8.flatten(), bins=256, range=(0, 256), density=False)
    plt.title("result8")
    plt.xlabel("grayscale")
    plt.ylabel("density")
    plt.savefig(output_result8_hist)
    plt.show();

    sample3 = cv2.imread(psample3, cv2.IMREAD_GRAYSCALE)
    plt.hist(sample3.flatten(), bins=256, range=(0, 256), density=False)
    plt.title("sample3")
    plt.xlabel("grayscale")
    plt.ylabel("density")
    plt.savefig(output_sample3_hist)
    plt.show();

    result9 = cv2.imread(presult9, cv2.IMREAD_GRAYSCALE)
    plt.hist(result9.flatten(), bins=256, range=(0, 256), density=False)
    plt.title("result9")
    plt.xlabel("grayscale")
    plt.ylabel("density")
    plt.savefig(output_result9_hist)
    plt.show();
    '''






