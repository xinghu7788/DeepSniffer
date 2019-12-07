#essemble attack eg two adversary models
import torch
import torch.nn as nn
from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
from models import *
import sys
import random


victim_model = resnet18('checkpoint/resnet18_web.pth.tar').cuda()

#vgg


class Attack():

    
    criterion = nn.CrossEntropyLoss().cuda()
    e = 0.05
    succ = 0

    def ensemble_attack1(self, model_name, target, org_target):
        print(model_name)
        image_arr = torch.load('data_100/class_'+ str(org_target) +'.pth.tar')
        victim_model.eval()
        fake_label = torch.LongTensor(1)
        fake_label[0] = target
        fake_label = torch.autograd.Variable(fake_label.cuda(), requires_grad=False)
        org_label = torch.zeros(image_arr.shape[0])
        attack_label = torch.zeros(image_arr.shape[0])
        succ_label = torch.zeros(image_arr.shape[0])
        succ_iter = torch.zeros(image_arr.shape[0])
        
        if model_name == 'true_resnet':
            model1 = resnet18('./checkpoint_true_resnet/models/true_res18_v1.pth.tar')
            model2 = resnet18('./checkpoint_true_resnet/models/true_res18_v2.pth.tar')
            model3 = resnet18('./checkpoint_true_resnet/models/true_res18_v3.pth.tar')
            model4 = resnet18('./checkpoint_true_resnet/models/true_res18_v4.pth.tar')
            models = [model1.cuda(), model2.cuda(), model3.cuda(), model4.cuda()] 
        
        elif model_name == 'vgg':
            model1 = vgg11('checkpoint/vgg11_web.pth.tar').cuda()
            model2 = vgg13('checkpoint/vgg13_web.pth.tar').cuda()
            model3 = vgg16('checkpoint/vgg16_web.pth.tar').cuda()
            model4 = vgg19('checkpoint/vgg19_web.pth.tar').cuda()
            models = [model1.cuda(), model2.cuda(), model3.cuda(), model4.cuda()]
        elif model_name == 'resnet':
            model1 = resnet34('checkpoint/resnet34_web.pth.tar').cuda()
            model2 = resnet50('checkpoint/resnet50_web.pth.tar').cuda()
            model3 = resnet101('checkpoint/resnet101_web.pth.tar').cuda()
            model4 = resnet152('checkpoint/resnet152_web.pth.tar').cuda()
            models = [model1.cuda(), model2.cuda(), model3.cuda(), model4.cuda()]

        elif model_name == 'mix':
            model1 = alexnet('checkpoint/alexnet_web.pth.tar').cuda()
            model2 = inception_v3('checkpoint/inception_v3_web.pth.tar').cuda()
            model3 = squeezenet1_0('checkpoint/squeezenet1_0_web.pth.tar').cuda()
            model4 = densenet121('checkpoint/densenet121_web.pth.tar').cuda()
            models = [model1.cuda(), [model2.cuda()], model3.cuda(), model4.cuda()]

        elif model_name == 'densenet':
            model1 = densenet121('checkpoint/densenet121_web.pth.tar').cuda()
            model2 = densenet161('checkpoint/densenet161_web.pth.tar').cuda()
            model3 = densenet169('checkpoint/densenet169_web.pth.tar').cuda()
            model4 = densenet201('checkpoint/densenet201_web.pth.tar').cuda()
            models = [model1.cuda(), model2.cuda(), model3.cuda(), model4.cuda()]

        elif model_name == 'deepsniffer':
            model1 = fake_1('checkpoint_deepsniffer/fake1_v1.pth.tar').cuda()
            model2 = fake_4('checkpoint_deepsniffer/fake4.pth.tar').cuda()
            model3 = fake_2('checkpoint_deepsniffer/fake2.pth.tar').cuda()
            model4 = fake_5('checkpoint_deepsniffer/fake5.pth.tar').cuda()
            models = [model1.cuda(), model2.cuda(), model3.cuda(), model4.cuda()]

        diff_succ = 0.0
        diff_all  = 0.0

        for i in range(image_arr.shape[0]):
        #for i in range(2):
            org_image = torch.FloatTensor(1, 3, 224, 224)
            org_image[0] = image_arr[i]
            org_image = torch.autograd.Variable(org_image.cuda(), requires_grad=True)

            output = victim_model(org_image)
            _, org_pred = output.topk(1, 1, True, True)
            org_pred = org_pred.data[0, 0]

            fake_image = org_image.clone()

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

                    if (i + 1) % 20 == 0:
                        print(str(i) + '\torg prediction: ' + str(org_pred.item()) + '\tfake prediction: ' + str(fake_pred.item()) +
                          '\tattack_pred: ' + str(attack_pred_list) + '\te: ' + str(self.e) + '\titer: ' + str(iter) + '\tsucc: ' + str(self.succ))

                    org_label[i] = org_pred.item()
                    attack_label[i] = fake_pred.item()
                    succ_iter[i] = iter + 1
                    
                    diff = torch.sum((org_image - fake_image) ** 2).item()
                    diff_all += diff

                    if fake_label.item() == fake_pred:
                        diff_succ += diff
                        self.succ += 1
                        succ_label[i] = 1
                    break


        diff_all /= (1.0 * image_arr.shape[0])
        if self.succ > 0:
            diff_succ /= (1.0 * self.succ)
        #print('total: ' + str(i + 1) + '\tsuccess: ' + str(self.succ) + '\tavg iter: ' + str(torch.mean(succ_iter).item()))
        
        str_log = 'src: ' + str(org_target) + '\ttar: ' + str(target)+ '\ttotal: ' + str(i + 1) + '\tsuccess: ' + str(self.succ) + '\tavg iter: ' + str(torch.mean(succ_iter).item()) + '\tdiff_suc: ' + str(diff_succ) + '\tdif_total: ' + str(diff_all) 
        print(str_log)
        str_filename = './rebuttle_result/' + model_name + '.log'
        f = open(str_filename, "a")
        f.write(str_log)
        f.write("\n")
        f.close()
        self.succ = 0

        # torch.save({'org_label': org_label, 'attack_label': attack_label, 'succ_label': succ_label, 'succ_iter': succ_iter},
        #         './result_100/' + model_name + '_' + str(int(org_label[0].item())) +  '_' + str(target) + '.pth.tar')



def main():

    #model_name = sys.argv[1]
    model_list_name = ['deepsniffer','vgg','resnet','mix','densenet']
    #model_list_name = ['our2']
    #model_list_name = ['true_resnet']
    attack = Attack()
    history_list = []
    random.seed(1)
    for temporal in range(1,51):
        org_target = random.randint(1, 998)
        if org_target in history_list:
            temporal = temporal - 1
            continue
        attack_target = (org_target + 5)%999
        history_list.append(temporal)
        print(temporal, attack_target)
        for model_name in model_list_name:
            attack.ensemble_attack1(model_name, attack_target, org_target)
            
                
    #attack.ensemble_attack1(model_list_name,attack_target)
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--base_dir', default=os.getcwd())
    #parser.add_argument('--input', default='sample_generator/test_sample_data')
    #args = parser.parse_args()
    #record_kernels(args)
    #print(unknown_kernel_dict)

if __name__ == '__main__':
    main()










