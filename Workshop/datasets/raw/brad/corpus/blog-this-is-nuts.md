# This is Nuts

---

Yesterday the New York Times ran an Op-Ed piece that led with the collapse of the effort to create a hand-held device for managing the 2010 census.


> The latest problem is the Census Bureau’s failure— after nearly four years and almost $600 million — to develop a reliable hand-held computer system for counting millions of Americans who are not counted by mail. Census takers will now have to use far less accurate paper and pencil.


The latest problem is the Census Bureau’s failure— after nearly four years and almost $600 million — to develop a reliable hand-held computer system for counting millions of Americans who are not counted by mail. Census takers will now have to use far less accurate paper and pencil.

The full NYT Opinion piece ishere. The AP story ishere.

This is more than a technology problem. It’s a colossal screw up. But there is an underlying technology problem. There is no easy way to create a purpose built device and integrate it into a new or existing process. The current method requires that the entire device be designed from scratch, all of the components or subsystems sourced anew for each new design. Finally you have to write custom software from scratch to stitch all the components together. It can take months to get a prototype to boot and years to integrate everything into a working product – electrical, mechanical, industrial engineering, manufacturing process engineering, QA, and support. And this is just the device. You then have to integrate that device into a business process and software applications environemnt. Half the time, by the time you are done, the process has changed and the technology embedded in the device is obsolete.

Contrast that to the way web services are built today. Start with the open source LAMP stack, modify slightly for your unique requirements, cut and paste a little Java script to mash up two or three other services on the web and than spend a couple of weeks hacking in a light weight scripting language like Ruby or PHP and presto you have a service that can serve hundreds of thousands of users. With a little more work it can support millions. The foundation of open source software and standard interfaces makes it much easier to create an innovative service and get it into the market quickly and cheaply.

Many of the folks who follow Bug Labs are really jazzed by the potential to scratch their own itch – to create a quirky device that meets a personal need. I wrote about my Bughere. It is great that Bug has captured the imagination of hardware hackers, but Bug is so much more than a modern day Heathkit.

Bugs goal is to create an architecture that would allow anyone (even the Census Bureau) to quickly build a device and integrate it into a service. I am not going to try to design a device for the Census Bureau here, but you could have fun mixing and matching a video camera, a GPS, a touch screen, WIFI and any number of other components to create a device. That, however, is the easy part. The hard part is making all of those components work together in an application. What if you wanted to be able to extract census data from a video interview? How would you pick out an address on a mailbox or a front door and cross check that address with the on board GPS, how could use voice recognition software to create a transcript of the interview and identify key elements of the census questionnaire and correctly populate the a database. Maybe this is a really dumb way to automate a census. That is not the point. The point is that the Bug Labs architecture lowers the cost of failure for the Census Bureau, not just because they can iterate cheaply as they define and then refine their device, but also because all the work they do on the device and on the software is reusable. So even if they end up deciding that their first hardware, software, and service implementation is fatally flawed, they can tweak a little here and there, evolving all three together until they get it right.

That makes a heck of a lot more sense than authorizing another $600mm and sending them off to try again the old way.
