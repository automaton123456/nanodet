import torch
import torch.nn as nn
import math

def conv_1x1_bn(inp, oup, stride, activation):
    total_padding = (1 - 1) // 2
  
    if activation == True:
      return nn.Sequential(
          nn.Conv2d(inp, oup, 1, stride, total_padding, bias=False),
          nn.BatchNorm2d(oup),
          nn.ReLU6()
      ) 
    else:  
      return nn.Sequential(
          nn.Conv2d(inp, oup, 1, stride, total_padding, bias=False),
          #nn.BatchNorm2d(oup),
      )


def conv_3x3_bn(inp, oup, stride, activation):
    total_padding = (3 - 1) // 2

    if activation == True:
      return nn.Sequential(
          nn.Conv2d(inp, oup, 3, stride, total_padding, bias=False),
          nn.BatchNorm2d(oup),
          nn.ReLU6()
      )
    else:
      return nn.Sequential(
          nn.Conv2d(inp, oup, 3, stride, total_padding, bias=False),
          #nn.BatchNorm2d(oup)
      )    

def conv_5x5_bn(inp, oup, stride, activation):
    total_padding = (5 - 1) // 2

    if activation == True:
      return nn.Sequential(
          nn.Conv2d(inp, oup, 5, stride, total_padding, bias=False),
          nn.BatchNorm2d(oup),
          nn.ReLU6()
      )
    else:    
      return nn.Sequential(
          nn.Conv2d(inp, oup, 5, stride, total_padding, bias=False),
          #nn.BatchNorm2d(oup),
      )

def depth_bn(inp, oup, stride, activation):
    total_padding = (3 - 1) // 2
    
    if activation == True:
      return nn.Sequential( 
          nn.Conv2d(inp, oup, 3, stride, total_padding, bias=False, groups = inp),
          nn.BatchNorm2d(oup),
          nn.ReLU6()
      ) 
    
    else:
      return nn.Sequential( 
          nn.Conv2d(inp, oup, 3, stride, total_padding, bias=False, groups = inp),
          #nn.BatchNorm2d(oup)
      )      
    
class MobileNetEdgeV21080(nn.Module):    
  def __init__(self, model_name, out_stages=(1, 3, 5), activation="ReLU6", pretrain=True):
    super(MobileNetEdgeV21080, self).__init__()

    self.stem = nn.Sequential(
        conv_5x5_bn(3, 32, 2, True),
        conv_1x1_bn(32, 16, 1, False),
        conv_3x3_bn(16, 128, 2, True),
        conv_1x1_bn(128, 32, 1, False)
    )

    self.b0_0 = conv_3x3_bn(16, 64, 1, True)
    self.b0_1 = conv_3x3_bn(16, 64, 1, True)
    self.b0_2 = conv_1x1_bn(128, 32, 1, False)

    self.b1_0 = conv_3x3_bn(8, 64, 2, True)
    self.b1_1 = conv_3x3_bn(8, 64, 2, True)
    self.b1_2 = conv_3x3_bn(8, 64, 2, True)
    self.b1_3 = conv_3x3_bn(8, 64, 2, True)

    self.b2_0 = conv_1x1_bn(256, 48, 1, False)
    self.b2_1 = conv_3x3_bn(16, 64, 1, True)
    self.b2_2 = conv_3x3_bn(16, 64, 1, True)
    self.b2_3 = conv_3x3_bn(16, 64, 1, True)
    self.b2_4 = conv_1x1_bn(192, 48, 1, False)

    #self.b3_0 = conv_1x1_bn(256, 48, 1, False)
    self.b3_1 = conv_3x3_bn(16, 64, 1, True)
    self.b3_2 = conv_3x3_bn(16, 64, 1, True)
    self.b3_3 = conv_3x3_bn(16, 64, 1, True)
    self.b3_4 = conv_1x1_bn(192, 48, 1, False)

    #self.b4_0 = conv_1x1_bn(256, 48, 1, False)
    self.b4_1 = conv_3x3_bn(16, 64, 1, True)
    self.b4_2 = conv_3x3_bn(16, 64, 1, True)
    self.b4_3 = conv_3x3_bn(16, 64, 1, True)
    self.b4_4 = conv_1x1_bn(192, 48, 1, False)    

    self.b5_0 = conv_3x3_bn(8, 64, 2, True)
    self.b5_1 = conv_3x3_bn(8, 64, 2, True)
    self.b5_2 = conv_3x3_bn(8, 64, 2, True)
    self.b5_3 = conv_3x3_bn(8, 64, 2, True)
    self.b5_4 = conv_3x3_bn(8, 64, 2, True)
    self.b5_5 = conv_3x3_bn(8, 64, 2, True)
    self.b5_6 = conv_1x1_bn(384, 80, 1, False)

    self.b6_0 = conv_1x1_bn(80, 320, 1,True)
    self.b6_1 = depth_bn(320, 320, 1,True)
    self.b6_2 = conv_1x1_bn(320, 80, 1, False)

    self.b7_0 = conv_1x1_bn(80, 320, 1,True)
    self.b7_1 = depth_bn(320, 320, 1,True)
    self.b7_2 = conv_1x1_bn(320, 80, 1, False)

    self.b8_0 = conv_1x1_bn(80, 320, 1,True)
    self.b8_1 = depth_bn(320, 320, 1,True)
    self.b8_2 = conv_1x1_bn(320, 80, 1, False)

    self.b9_0 = conv_1x1_bn(80, 640, 1, True)
    self.b9_1 = depth_bn(640, 640, 1, True)
    self.b9_2 = conv_1x1_bn(640, 112, 1, False)

    self.b10_0 = conv_1x1_bn(112, 448, 1, True)
    self.b10_1 = depth_bn(448, 448, 1, True)
    self.b10_2 = conv_1x1_bn(448, 112, 1, False)

    self.b11_0 = conv_1x1_bn(112, 448, 1, True)
    self.b11_1 = depth_bn(448, 448, 1, True)
    self.b11_2 = conv_1x1_bn(448, 112, 1, False)

    self.b12_0 = conv_1x1_bn(112, 448, 1, True)
    self.b12_1 = depth_bn(448, 448, 1, True)
    self.b12_2 = conv_1x1_bn(448, 112, 1, False)    

    self.b13_0 = conv_1x1_bn(112, 896, 1, True)
    self.b13_1 = depth_bn(896, 896, 2, True)
    self.b13_2 = conv_1x1_bn(896, 160, 1, False)

    self.b14_0 = conv_1x1_bn(160, 640, 1, True)
    self.b14_1 = depth_bn(640, 640, 1, True)
    self.b14_2 = conv_1x1_bn(640, 160, 1, False)

    self.b15_0 = conv_1x1_bn(160, 640, 1, True)
    self.b15_1 = depth_bn(640, 640, 1, True)
    self.b15_2 = conv_1x1_bn(640, 160, 1, False)

    self.b16_0 = conv_1x1_bn(160, 640, 1, True)
    self.b16_1 = depth_bn(640, 640, 1, True)
    self.b16_2 = conv_1x1_bn(640, 160, 1, False)


    self.b17_0 = conv_1x1_bn(160, 1280, 1, True)
    self.b17_1 = depth_bn(1280, 1280, 1, True)
    self.b17_2 = conv_1x1_bn(1280, 192, 1, False)
    self.b17_3 = conv_1x1_bn(192, 1280, 1, True)

    self._initialize_weights()


  def forward(self, x): 
      output = []
      x = self.stem(x)
            
      #Block
      (s1,s2) = torch.split(x, 16, 1)

      s1 = self.b0_0(s1)
      s2 = self.b0_1(s2)

      b = torch.cat((s1,s2), 1)
      b = self.b0_2(b)
      x = torch.add(x,b)

      #Block 1
      (s1,s2,s3,s4) = torch.split(x, 8, 1)
      
      s1 = self.b1_0(s1)
      s2 = self.b1_1(s2)
      s3 = self.b1_2(s3)
      s4 = self.b1_3(s4)
      x = torch.cat((s1,s2,s3,s4), 1)    
      
      #Block 2  
      x = self.b2_0(x) 
      
      (s1,s2,s3) = torch.split(x, 16, 1)
      
      s1 = self.b2_1(s1)
      s2 = self.b2_2(s2)
      s3 = self.b2_3(s3)

      b = torch.concat((s1,s2,s3), 1)
      
      b = self.b2_4(b)
      
      x = torch.add(x,b)

      
      #Block 3       
      (s1,s2,s3) = torch.split(x, 16, 1)
      
      s1 = self.b3_1(s1)
      s2 = self.b3_2(s2)
      s3 = self.b3_3(s3)

      b = torch.concat((s1,s2,s3), 1)
      
      b = self.b3_4(b)
      
      x = torch.add(x,b)


      #Block 4  
      (s1,s2,s3) = torch.split(x, 16, 1)
      
      s1 = self.b4_1(s1)
      s2 = self.b4_2(s2)
      s3 = self.b4_3(s3)

      b = torch.concat((s1,s2,s3), 1)
      
      b = self.b4_4(b)
      
      x = torch.add(x,b)

      #print("Level 1 " + str(x.size()))
      output.append(x)

      #Block 5
      #Split into 6 bits
      (s1,s2,s3,s4,s5,s6) = torch.split(x, 8, 1)

      s1 = self.b5_0(s1)
      s2 = self.b5_1(s2)
      s3 = self.b5_2(s3)
      s4 = self.b5_3(s4)
      s5 = self.b5_4(s5)
      s6 = self.b5_5(s6)
      x = torch.concat((s1,s2,s3,s4,s5,s6), 1)

      x = self.b5_6(x)



      #Block 6
      b = self.b6_0(x)
      b = self.b6_1(b)
      b = self.b6_2(b)
      x = torch.add(x,b)

      #Block 7
      b = self.b7_0(x)
      b = self.b7_1(b)
      b = self.b7_2(b)
      x = torch.add(x,b)

      #Block 8
      b = self.b8_0(x)
      b = self.b8_1(b)
      b = self.b8_2(b)
      x = torch.add(x,b)
      
      #Block 9
      x = self.b9_0(x)
      x = self.b9_1(x)
      x = self.b9_2(x)

      #Block 10
      b = self.b10_0(x)
      b = self.b10_1(b)
      b = self.b10_2(b)
      x = torch.add(x,b)

      #Block 11
      b = self.b11_0(x)
      b = self.b11_1(b)
      b = self.b11_2(b)
      x = torch.add(x,b)

      #Block 12
      b = self.b12_0(x)
      b = self.b12_1(b)
      b = self.b12_2(b)
      x = torch.add(x,b) 

      #print("Level 2 " + str(x.size()))
      output.append(x)

      #Block 13
      x = self.b13_0(x)
      x = self.b13_1(x)
      x = self.b13_2(x)
      
      #Block 14
      b = self.b14_0(x)
      b = self.b14_1(b)
      b = self.b14_2(b)
      x = torch.add(x, b)



      #Block 15
      b = self.b15_0(x)
      b = self.b15_1(b)
      b = self.b15_2(b)
      x = torch.add(x, b)

      #Block 16
      b = self.b16_0(x)
      b = self.b16_1(b)
      b = self.b16_2(b)
      x = torch.add(x, b)

      #Block 17
      x = self.b17_0(x)
      x = self.b17_1(x)
      x = self.b17_2(x)
      x = self.b17_3(x)
      #print("Level 3 " + str(x.size()))
      output.append(x)
  
      return output #x

  def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()
    
  def load_pretrain(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=True)




#net = Net()
#print(net) 

#model_scripted = torch.jit.script(net) # Export to TorchScript
#model_scripted.save('/content/model_scripted.pt') # Save

#image = torch.randn((1,3,320,320))

#model = Net("")
#test = model(image)
#print(len(test))
#torch.onnx.export(model, image, "super_resolution.onnx", opset_version=10)
