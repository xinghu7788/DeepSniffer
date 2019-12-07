#essemble attack eg two adversary models
import torch
import torch.nn as nn
import sys
import random
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable, grad
from skimage.io import imsave
from models import *



victim_model = resnet18('checkpoint/resnet18_web.pth.tar').cuda()
#model1 = fake_1('checkpoint_substitute/fake1_v1.pth.tar').cuda()
#model2 = fake_0('checkpoint_substitute/fake0_res18.pth.tar').cuda()
#model3 = fake_01('checkpoint_substitute/fake01.pth.tar').cuda()
#model4 = vgg11('checkpoint/vgg11_web.pth.tar').cuda()
#model5 = vgg13('checkpoint/vgg13_web.pth.tar').cuda()
#model6 = vgg16(('checkpoint/vgg16_web.pth.tar').cuda()
#model7 = vgg19(('checkpoint/vgg19_web.pth.tar').cuda()
#model8 = vgg11_bn('checkpoint/vgg11_bn_web.pth.tar').cuda()
#model9 = vgg13('checkpoint/vgg13_bn_web.pth.tar').cuda()
#model10 = vgg16(('checkpoint/vgg16_bn_web.pth.tar').cuda()
#model11 = vgg19(('checkpoint/vgg19_bn_web.pth.tar').cuda()
#models = [model1, model3,model4,model2] #, model3, model4]
#model_list_name = ["densenet121","densenet161","squeezenet1_0","squeezenet1_1"]
#model_complete_list_name = ["vgg11","vgg13","vgg16","resnet34"]

org_target = str(int(sys.argv[1]) + 1)  # the original label
model_complete_list_name = ["vgg11","vgg13","vgg16","vgg19","resnet34","resnet50","resnet101","resnet152","densenet121","densenet161","densenet169","densenet201","squeezenet1_0","squeezenet1_1","alexnet"]
#model_complete_list_name = ["vgg11","vgg13","vgg16","vgg19","resnet34","resnet50","resnet101","resnet152","densenet121","densenet161","densenet169","densenet201","squeezenet1_0","squeezenet1_1","alexnet","inception_v3"]
class Attack():


    image_arr = torch.load('data_100/class_'+ org_target +'.pth.tar')
    criterion = nn.CrossEntropyLoss().cuda()
    e = 0.05
    succ = 0
    #ensemble target
    def ensemble_models(self,model_name_list):
        models = []
        print('model_name_list')
        print(model_name_list)
        for model_name in model_name_list:
            models.append(self.generate_model(model_name))
        return models
        
    
    def generate_model(self, model_name):
        if model_name =="vgg11":
            model = vgg11('checkpoint/vgg11_web.pth.tar').cuda()
        elif model_name == "vgg13":
            model = vgg13('checkpoint/vgg13_web.pth.tar').cuda()
        elif model_name == "vgg16":
            model = vgg16('checkpoint/vgg16_web.pth.tar').cuda()
        elif model_name == "vgg19":
            model = vgg19('checkpoint/vgg19_web.pth.tar').cuda()
        elif model_name == "resnet18":
            model = resnet18('checkpoint/resnet18_web.pth.tar').cuda() 
        elif model_name == "resnet34":
            model = resnet34('checkpoint/resnet34_web.pth.tar').cuda()
        elif model_name == "resnet50":
            model = resnet50('checkpoint/resnet50_web.pth.tar').cuda()
        
        elif model_name == "resnet101":
            model = resnet101('checkpoint/resnet101_web.pth.tar').cuda()
        
        elif model_name == "resnet152":
            model = resnet152('checkpoint/resnet152_web.pth.tar').cuda()
        
        elif model_name == "densenet121":
            model = densenet121('checkpoint/densenet121_web.pth.tar').cuda()
        
        elif model_name == "densenet161":
            model = densenet161('checkpoint/densenet161_web.pth.tar').cuda()
        elif model_name == "densenet169":
            model = densenet169('checkpoint/densenet169_web.pth.tar').cuda()
        elif model_name == "densenet201":
            model = densenet201('checkpoint/densenet201_web.pth.tar').cuda()
        
        elif model_name == "squeezenet1_0":
            model = squeezenet1_0('checkpoint/squeezenet1_0_web.pth.tar').cuda()
        elif model_name == "squeezenet1_1":
            model = squeezenet1_1('checkpoint/squeezenet1_1_web.pth.tar').cuda()
        
        elif model_name == "alexnet":
            model = alexnet('checkpoint/alexnet_web.pth.tar').cuda()
        elif model_name == "inception_v3":
            model = inception_v3('checkpoint/inception_v3_web.pth.tar').cuda()
        else:
            print("no such a model")
            exit(0)
        return model


    def ensemble_attack1(self, model_name_list, target):
        models = self.ensemble_models(model_name_list)
        victim_model.eval()
        fake_label = torch.LongTensor(1)
        fake_label[0] = target
        fake_label = torch.autograd.Variable(fake_label.cuda(), requires_grad=False)
        org_label = torch.zeros(self.image_arr.shape[0])
        attack_label = torch.zeros(self.image_arr.shape[0])
        succ_label = torch.zeros(self.image_arr.shape[0])
        succ_iter = torch.zeros(self.image_arr.shape[0])
        model_name = model_name_list[0] + ' '+ model_name_list[1] + ' ' + model_name_list[2] + ' ' + model_name_list[3] + '\t'
        self.succ = 0

        for i in range(self.image_arr.shape[0]):
        #for i in range(2):
            org_image = torch.FloatTensor(1, 3, 224, 224)
            org_image[0] = self.image_arr[i]
            org_image = torch.autograd.Variable(org_image.cuda(), requires_grad=True)

            output = victim_model(org_image)
            _, org_pred = output.topk(1, 1, True, True)
            org_pred = org_pred.data[0, 0]

            fake_image = org_image

            self.e = 0.04 / len(models)
            #modify the original image
            max_val = torch.max(org_image).item()
            min_val = torch.min(org_image).item()
            for iter in range(50):
                # calculate gradient
                grad = torch.zeros(1, 3, 224, 224).cuda()
                fake_image = torch.autograd.Variable(fake_image.cuda(), requires_grad=True)
                for m in models:
                    if type(m) == type([]):
                        m = m[0]
                        m.eval()
                        fake_image_ = F.upsample(fake_image, size=299, mode='bilinear')
                        output = m(fake_image_)
                        loss = self.criterion(output, fake_label)
                        loss.backward()
                        # print(loss)
                        grad += F.upsample(torch.sign(fake_image.grad), size=224, mode='bilinear')
                    else:
                        m.eval()
                        zero_gradients(fake_image)
                        output = m(fake_image)
                        loss = self.criterion(output, fake_label)
                        loss.backward()
                        #print(loss)
                        grad += torch.sign(fake_image.grad)

                fake_image = fake_image - grad * self.e
                fake_image[fake_image > max_val] = max_val
                fake_image[fake_image < min_val] = min_val
                output = victim_model(fake_image)

                _, fake_pred = output.topk(1, 1, True, True)
                fake_pred = fake_pred.data[0, 0]

                if fake_label.item() == fake_pred or iter == 49:
                    attack_pred_list = []
                    for m in models:
                        if type(m) == type([]):
                            output = m[0](F.upsample(fake_image, size=299, mode='bilinear'))
                        else:
                            output = m(fake_image)
                        _, attack_pred = output.topk(1, 1, True, True)
                        attack_pred_list.append(attack_pred.data[0, 0].item())

                    print(str(i) + '\torg prediction: ' + str(org_pred.item()) + '\tfake prediction: ' + str(fake_pred.item()) +
                          '\tattack_pred: ' + str(attack_pred_list) + '\te: ' + str(self.e) + '\titer: ' + str(iter))

                    org_label[i] = org_pred.item()
                    attack_label[i] = fake_pred.item()
                    succ_iter[i] = iter + 1

                    if fake_label.item() == fake_pred:
                        self.succ += 1
                        succ_label[i] = 1
                        print('succ: ' + str(self.succ))
                    break
        str_log = model_name + '\t' + 'total: ' + str(i + 1) + '\tsuccess: ' + str(self.succ) + '\tavg iter: ' + str(torch.mean(succ_iter).item())
        print(str_log)
        f = open("demofile_random.txt", "a")
        f.write(str_log)
        f.write("\n")
        torch.save({'org_label': org_label, 'attack_label': attack_label, 'succ_label': succ_label, 'succ_iter': succ_iter},
                   './result_100/' + model_name + '_' + str(int(org_label[0].item())) +  '_' + str(target) + '.pth.tar')
        self.succ = 0
        


def main():

    #model_name = sys.argv[1]
    model_list_name = []
    history_list = []


    
    attack_target = int(sys.argv[2])
    attack = Attack()
    for temporal in range(0,200):
        model_list_name = random.sample(model_complete_list_name,4)
        if model_list_name in history_list:
            temporal = temporal - 1
            continue
        print(model_list_name)
        attack.ensemble_attack1(model_list_name, attack_target)
        history_list.append(model_list_name)
    
    #attack.ensemble_attack1(model_list_name,attack_target)
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--base_dir', default=os.getcwd())
    #parser.add_argument('--input', default='sample_generator/test_sample_data')
    #args = parser.parse_args()
    #record_kernels(args)
    #print(unknown_kernel_dict)

if __name__ == '__main__':
    main()




