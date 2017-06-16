#!/usr/bin/perl


print "\@relation whatever\n";
my @lines = <STDIN>;

my $line = $lines[0];



chomp($line);

my @features = split(",", $line);
my $idx_label = $#features;

for my $idx (0..$#features-1)
{
    print "\@attribute $features[$idx] numeric\n";
}

my $label_name = $features[$#features];


my %label_set;
my $cut_lable_pos = $idx_label + 1;
#print $cut_lable_pos;


#    print @labels;
foreach my $idx (1 .. $#lines)
{
    my $l1 = $lines[$idx];
    chomp($l1);
    my @arr_l1 = split(",", $l1);
    my $itr = $arr_l1[$idx_label];

    $label_set{$itr} = 1;
}

print "\@attribute $label_name ";
print "{";
print join(",", keys %label_set);
print "}\n";




print "\@data\n";
foreach my $idx (1 .. $#lines)
{
    print "$lines[$idx]";

}



